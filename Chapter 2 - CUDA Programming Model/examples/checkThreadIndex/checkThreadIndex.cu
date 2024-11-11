#include <cuda_runtime.h>
#include <stdio.h>

/**
 * This function checks the error of the CUDA runtime API functions
 * @param call The CUDA runtime API function to check the error
 */
#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10*error); \
    } \
}

/**
 * This function initializes the array with the value of the index
 * @param ip The pointer to the array
 * @param size The size of the array
 */
void initialInt(int *ip, int size) {
    for (int i=0; i<size; i++) {
        ip[i] = i;
    }
}

/**
 * This function prints the matrix
 * @param C The pointer to the matrix
 * @param nx The number of columns of the matrix
 * @param ny The number of rows of the matrix
 */
void printMatrix(int *C, const int nx, const int ny) {
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);
    for (int iy=0; iy<ny; iy++) {
        for (int ix=0; ix<nx; ix++) {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}


/**
 * This function prints the thread index
 * @param A The pointer to the matrix
 * @param nx The number of columns of the matrix
 * @param ny The number of rows of the matrix
 */
__global__ void printThreadIndex(int *A, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    printf("ThreadIdx(%d, %d) BlockIdx(%d, %d) Coordinate(%d, %d) GlobalIdx %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx);
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    //get the device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    // malloc host memory
    int *h_A;
    h_A = (int *)malloc(nBytes);

    // initialize host matrix with integer
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // malloc device memory
    int *d_MatA;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up execution configuration
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // invoke the kernel
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    CHECK(cudaDeviceSynchronize());

    // free host and device memory
    CHECK(cudaFree(d_MatA));
    free(h_A);

    // reset device
    CHECK(cudaDeviceReset());
    return 0;
}