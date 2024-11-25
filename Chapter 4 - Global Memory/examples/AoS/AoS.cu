#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define LEN 1<<20

struct innerStruct {
    float x;
    float y;
};

/**
    * @brief Get the current time in seconds
*/
double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}


/**
    * @brief Initialize the innerStruct array
    * 
    * @param ip innerStruct array
    * @param size size of the array
    *   
    * @return void
*/
void initialInnerStruct(innerStruct *ip, int size) {
    for (int i=0; i<size; i++) {
        ip[i].x = (float)(rand() & 0xFF) / 10.0f;
        ip[i].y = (float)(rand() & 0xFF) / 10.0f;
    }
}

/**
    * @brief Kernel function to add 10 to x and 20 to y
    * 
    * @param data input innerStruct array
    * @param result output innerStruct array
    * @param n size of the array
    *   
    * @return void
*/
__global__ void testInnerStruct(innerStruct *data,
    innerStruct *result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

/**
    * @brief Host function to add 10 to x and 20 to y
    * 
    * @param data input innerStruct array
    * @param result output innerStruct array
    * @param n size of the array
    *   
    * @return void
*/
void testInnerStructHost(innerStruct *data, innerStruct *result, const int n) {
    for (int i=0; i<n; i++) {
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

/**
    * @brief warmup kernel function
    * @param data input innerStruct array
    * @param result output innerStruct array
    * @param n size of the array
*/
__global__ void warmup(innerStruct *data, innerStruct *result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerStruct tmp = data[i];
        result[i] = tmp;
    }
}

/**
    * @brief Check the result of the innerStruct array
    * 
    * @param A innerStruct array
    * @param B innerStruct array
    * @param n size of the array
    *   
    * @return void
*/
void checkInnerStruct(innerStruct *A, innerStruct *B, int n) {
    double epsilon = 1.0E-8;
    for (int i=0; i<n; i++) {
        if (abs(A[i].x - B[i].x) > epsilon) {
            printf("different on %dth element: %f, %f\n", i, A[i].x, B[i].x);
            return;
        }
        if (abs(A[i].y - B[i].y) > epsilon) {
            printf("different on %dth element: %f, %f\n", i, A[i].y, B[i].y);
            return;
        }
    }
    printf("check result: PASSED\n");
}


int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s test struct of array at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // allocate host memory
    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);
    innerStruct *h_A = (innerStruct *)malloc(nBytes);
    innerStruct *hostRef = (innerStruct *)malloc(nBytes);
    innerStruct *gpuRef = (innerStruct *)malloc(nBytes);
    // initialize host array

    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef,nElem);
    // allocate device memory
    innerStruct *d_A,*d_C;
    cudaMalloc((innerStruct**)&d_A, nBytes);
    cudaMalloc((innerStruct**)&d_C, nBytes);

    // copy data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    // set up offset for summary
    int blocksize = 128;
    if (argc>1) blocksize = atoi(argv[1]);

    // execution configuration
    dim3 block (blocksize,1);
    dim3 grid ((nElem+block.x-1)/block.x,1);

    // kernel 1: warmup
    double iStart = seconds();
    warmup <<< grid, block >>> (d_A, d_C, nElem);
    cudaDeviceSynchronize();
    double iElaps = seconds() - iStart;
    printf("warmup <<< %3d, %3d >>> elapsed %f sec\n",grid.x, block.x,iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);
    
    // kernel 2: testInnerStruct
    iStart = seconds();
    testInnerStruct <<< grid, block >>> (d_A, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("innerstruct <<< %3d, %3d >>> elapsed %f sec\n",grid.x,
    block.x,iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);

    // free memories both host and device
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);
    
    // reset device
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
