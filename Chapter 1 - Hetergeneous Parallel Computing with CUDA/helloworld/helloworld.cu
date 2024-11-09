#include <stdio.h>

__global__ void helloFromGPU(void){
    int threadId = threadIdx.x;
    printf("Hello World from GPU thread %d!\n", threadId);
}

int main(void)
{
    printf("Hello World from CPU\n");
    helloFromGPU<<<1, 1>>>();
    cudaDeviceReset();
    return 0;
}