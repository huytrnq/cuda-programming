#include <cuda_runtime.h>
#include <stdio.h>


/**
 * @brief Kernel function to check the index of a thread.
 *
 * This CUDA kernel function prints the index of a thread within a block and the index of the block
 * within the grid. It also prints the dimensions of the block and grid.
 */
__global__ void checkIndex(void)
{
    printf("threadIdx.x %d, threadIdx.y %d, threadIdx.z %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx.x %d, blockIdx.y %d, blockIdx.z %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("blockDim.x %d, blockDim.y %d, blockDim.z %d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim.x %d, gridDim.y %d, gridDim.z %d\n", gridDim.x, gridDim.y, gridDim.z);
}

/**
 * @brief Main function.
 *
 * This function is the entry point for the program. It initializes an array of data on the host, and
 * then copies it to the device. The sumArraysOnDevice kernel is then launched to compute the sum of
 * the arrays in parallel on the device. The result is copied back to the host and printed to the
 * console.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line argument strings.
 * @return An integer value indicating the result of program execution.
 */
int main(int argc, char **argv)
{
    //definde total data element
    int nElem = 6;

    //define grid and block structure
    // define block with 3 threads, (1, 1, 1) is the default
    dim3 block (3);
    // define grid with 2 blocks, (1, 1, 1) is the default
    dim3 grid ((nElem + block.x - 1) / block.x);
    // The result is that there are 3 threads in each block and 2 blocks in the grid

    // check grid and block dimension from host side
    printf("grid.x %d, grid.y %d, grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d, block.y %d, block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    // reset device before you leave
    cudaDeviceReset();

    return(0);
}