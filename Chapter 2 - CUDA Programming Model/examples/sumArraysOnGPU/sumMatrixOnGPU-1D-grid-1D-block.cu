#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call) { \
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}\

/**
 * @brief Get the current time in seconds
 * 
 * @return double The current time in seconds
 */
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

/**
 * @brief Initialize the data on the host
 * @param ip The pointer to the data
 * @param size The size of the data
*/
void initializeData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

/**
 * @brief Sum the matrix on the host
 * @param A The first matrix pointer
 * @param B The second matrix pointer
 * @param C The result matrix pointer
 * @param nx The size of the x-axis
 * @param ny The size of the y-axis
*/
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++){
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        // move to next row by incrementing the pointer as the array is in row-major order
        ia += nx; 
        ib += nx;
        ic += nx;
    }
}

/**
    * @brief Sum the matrix on the device
    * @param MatA The first matrix pointer
    * @param MatB The second matrix pointer
    * @param MatC The result matrix pointer
    * @param nx The size of the x-axis
    * @param ny The size of the y-axis
*/
__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    // Get global index in a 2D grid
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx) {
        for (int iy = 0; iy < ny; iy++) {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

    
/**
 * @brief Check the result
 * @param hostRef The host result
 * @param gpuRef The GPU result
 * @param N The size of the arrays
*/
void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Argument values
 * @return int The exit status
*/
int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // set up data size of vectors
    int nx = 1<<14;
    int ny = 1<<14;

    // Calculate the size of the data
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    initializeData(h_A, nxy);
    initializeData(h_B, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix on the host side to provide a reference result
    double iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    double iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnHost elapsed %f sec\n", iElaps);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((float**)&d_MatA, nBytes));
    CHECK(cudaMalloc((float**)&d_MatB, nBytes));
    CHECK(cudaMalloc((float**)&d_MatC, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    // Set kernel to use 1D grid and 1D block
    // For this setup, the total number of threads is smaller than the total number of elements in the matrix
    // The kernel will be invoked multiple times to process all the elements which means that one thread will compute ny elements
    int dimx = 32;
    int dimy = 1;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x -1) / block.x, 1);

    iStart = cpuSecond();
    // invoke kernel at host side
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    //copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    //Check the result
    checkResult(hostRef, gpuRef, nxy);

    //free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    //free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    //reset device
    CHECK(cudaDeviceReset());
    return 0;
}