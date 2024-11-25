#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define LEN 1<<10

struct innerArray {
    float x[LEN];
    float y[LEN];
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
    * @brief Initialize the innerArray
    * 
    * @param ip innerArray
    * @param size size of the array
    *   
    * @return void
*/
void initialInnerArray(innerArray *ip, int size) {
    for (int i=0; i<size; i++) {
        ip->x[i] = (float)(rand() & 0xFF) / 10.0f;
        ip->y[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}


/**
    * @brief Kernel function to add 10 to x and 20 to y
    * 
    * @param data input innerArray
    * @param result output innerArray
    * @param n size of the array
    *   
    * @return void
*/
__global__ void testInnerArray(innerArray *data, innerArray *result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerArray tmp = * data;
        tmp.x[i] += 10.f;
        tmp.y[i] += 20.f;
        result->x[i] = tmp.x[i];
        result->y[i] = tmp.y[i];
    }
}

/**
    * @brief Host function to add 10 to x and 20 to y
    * 
    * @param data input innerArray
    * @param result output innerArray
    * @param n size of the array
    *   
    * @return void
*/
void testInnerArrayHost(innerArray *data, innerArray *result, const int n) {
    for (int i=0; i<n; i++) {
        innerArray tmp = *data;
        tmp.x[i] += 10.f;
        tmp.y[i] += 20.f;
        result->x[i] = tmp.x[i];
        result->y[i] = tmp.y[i];
    }
}


/**
    * @brief Check the result of the innerArray
    * @param A innerArray
    * @param B innerArray
    * @param n size of the array
*/
__global__ void warmup(innerArray *data, innerArray *result, const int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        innerArray tmp = *data;
        result->x[i] = tmp.x[i];
        result->y[i] = tmp.y[i];
    }
}

/**
    * @brief Check the result of the innerArray
    * @param A innerArray
    * @param B innerArray
    * @param n size of the array
*/
void checkInnerArray(innerArray *A, innerArray *B, int n) {
    double epsilon = 1.0E-8;
    for (int i=0; i<n; i++) {
        if (abs(A->x[i] - B->x[i]) > epsilon) {
            printf("different on %dth element: %f, %f\n", i, A->x[i], B->x[i]);
            return;
        }
        if (abs(A->y[i] - B->y[i]) > epsilon) {
            printf("different on %dth element: %f, %f\n", i, A->y[i], B->y[i]);
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
    size_t nBytes = nElem * sizeof(innerArray);
    innerArray *h_A = (innerArray *)malloc(nBytes);
    innerArray *hostRef = (innerArray *)malloc(nBytes);
    innerArray *gpuRef = (innerArray *)malloc(nBytes);
    // initialize host array

    initialinnerArray(h_A, nElem);
    testInnerArrayHost(h_A, hostRef, nElem);
    // allocate device memory
    innerArray *d_A,*d_C;
    cudaMalloc((innerArray**)&d_A, nBytes);
    cudaMalloc((innerArray**)&d_C, nBytes);

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
    checkInnerArray(hostRef, gpuRef, nElem);
    
    // kernel 2: testInnerArray
    iStart = seconds();
    testInnerArray <<< grid, block >>> (d_A, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("innerArray <<< %3d, %3d >>> elapsed %f sec\n",grid.x,
    block.x,iElaps);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerArray(hostRef, gpuRef, nElem);

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
