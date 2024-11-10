#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


/**
 * @brief Kernel function to sum two arrays on the device.
 *
 * This CUDA kernel function takes two input arrays A and B, and computes their element-wise sum,
 * storing the result in the output array C. The computation is performed in parallel using multiple
 * threads.
 *
 * @param A Pointer to the first input array.
 * @param B Pointer to the second input array.
 * @param C Pointer to the output array where the result will be stored.
 * @param N The number of elements in the arrays.
 */

__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief Print the contents of an array to the console.
 *
 * This function prints the contents of an array to the console, one element per line.
 *
 * @param data Pointer to the array to print.
 * @param size The number of elements in the array.
 */
void printData(float *data, const int size){
    for(int i = 0; i < size; i++){
        printf("%d: %0.2f\n", i, data[i]);
    }
}

/**
 * @brief Sum two arrays on the host.
 *
 * This function takes two input arrays A and B, and computes their element-wise sum, storing the
 * result in the output array C. The computation is performed in serial on the host.
 *
 * @param A Pointer to the first input array.
 * @param B Pointer to the second input array.
 * @param C Pointer to the output array where the result will be stored.
 * @param N The number of elements in the arrays.
 */
void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for(int i = 0; i < N; i++){
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief Initialize an array with random data.
 *
 * This function initializes an array with random floating-point values between 0 and 25.
 *
 * @param ip Pointer to the array to initialize.
 * @param size The number of elements in the array.
 */
void initialData(float *ip, int size){
    time_t t;
    srand((unsigned) time(&t));
    for(int i = 0; i < size; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

/**
 * @brief Main function.
 *
 * This function is the entry point for the program. It allocates memory on the host and device,
 * initializes arrays with random data, sums the arrays on the host and device, and prints the
 * results.
 */
int main(int argc, char **argv){
    // Set the number of elements in the arrays.
    int nElem = 10;
    // Calculate the number of bytes needed to store the arrays.
    size_t nBytes = nElem * sizeof(int);

    // Allocate memory on the host.
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    // Initialize the input arrays with random data.
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // Allocate memory on the device.
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // Copy the input arrays from the host to the device.
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // Sum the arrays on the host and device.
    sumArraysOnHost(h_A, h_B, h_C, nElem);
    printData(h_C, nElem);

    // Sum the arrays on the device with 1 block and 256 threads
    sumArraysOnDevice<<<1, 256>>>(d_A, d_B, d_C, nElem);
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
    printData(h_C, nElem);

    // Free memory on the host 
    free(h_A);
    free(h_B);
    free(h_C);

    // Free memory on the device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return(0);
}