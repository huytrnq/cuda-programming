#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK(call){\
    const cudaError_t error = call;\
    if (error != cudaSuccess){\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


/**
 * This kernel is used to warm up the GPU
 * @param c
 */
__global__ void warming_up(float *c){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = 100.0f;
    b = 200.0f;
    c[threadId] = a + b;
}

/**
 * This kernel demonstrates the divergence of warps
 * @param c
 */
__global__ void mathKernel1(float *c){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    // execute the if clause if the threadId is even, else execute the else clause
    // without considering the warp
    // gpu will check the condition for each thread as in this case if-else so there will be 2 independent warps
    if (threadId % 2 == 0){ 
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[threadId] = a + b;
}

/**
 * This kernel demonstrates hor to fix the divergence of warps
 * @param c
 */
__global__ void mathKernel2(float *c){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    // use even warps to execute the if clause else use odd warps
    if ((threadId / warpSize) % 2 == 0){
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[threadId] = a + b;
}

/**
 * This kernel demonstrates more divergence of warps
 * @param c
 */
__global__ void mathKernel3(float *c){
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    // more divergence
    // this will cause more divergence compared to kernel 1
    // both the conditions are if so all the threads will have to check the condition
    bool ipred = (threadId % 2 == 0);
    if (ipred){
        a = 100.0f;
    }
    if (!ipred){
        b = 200.0f;
    }
    
    c[threadId] = a + b;
}



int main(int argc, char **argv){
    //set up device
    int dev = 0;
    cudaDeviceProp cudaDeviceProp;
    cudaGetDeviceProperties(&cudaDeviceProp, dev);
    printf("Device %d: %s\n", dev, cudaDeviceProp.name);
    cudaSetDevice(dev);

    //set up data size
    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if  (argc > 2) size = atoi(argv[2]);
    printf("Data size %d ", size);

    //set up execution configuration
    dim3 block (blocksize, 1);
    dim3 grid (size + block.x -1 / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    //allocate gpu memory
    float *d_c;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc(&d_c, nBytes));

    //run a warm-up kernel to remove overhead
    double iStart, iElaps;
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    warming_up<<<grid, block>>>(d_c);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("warmup <<<%d, %d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    //run kernel 1
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_c);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<<%d, %d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    //run kernel 2
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(d_c);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("mathKernel2 <<<%d, %d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    //run kernel 3
    iStart = cpuSecond();
    mathKernel3<<<grid, block>>>(d_c);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("mathKernel3 <<<%d, %d>>> elapsed %f sec\n", grid.x, block.x, iElaps);

    cudaFree(d_c);
    cudaDeviceReset();
    return 0;
}