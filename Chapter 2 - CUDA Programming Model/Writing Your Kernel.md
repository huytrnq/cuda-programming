# Writing Your Kernel

- A kernel is a function that is executed on the device, you define the computation for a single thread, and the data access for that thread. When the kernel is called, many diffirent CUDA threads perform the same computation in parallel.
- A kernel must have a void return type.
- Function Type Qualifiers

    | Qualifier | Execution | Callable | Notes |
    |-----------|-----------|----------|-------|
    | `__global__` | Device | Host, Device of compute capability 3.0 and higher | Must have void return type |
    | `__device__` | Device | Callable from device only | |
    | `__host__` | Host | Callable from host only | Can be omitted |

- CUDA kernels are functions with restrictions:
    - Access to device memory only
    - Must have void return type
    - No support for a variable number of arguments
    - No support for static variables
    - No support for function pointers
    - Exhibit an asynchronous behavior

- As an illustration, consider a simple example of adding two vectors A and B of size N. The C code for vector addition on the host is given below:
    ```c    
        void sumArraysOnHost(float *A, float *B, float *C, const int N) {
            for (int i = 0; i < N; i++)
            C[i] = A[i] + B[i];
        }
    ```
- This a sequential code iterating N times, Peeling off the loop would produce the following kernel functions:
    ```c
        __global__ void sumArraysOnGPU(float *A, float *B, float *C) {
            int i = threadIdx.x;
            C[i] = A[i] + B[i];
        }
    ```
- You will notice that the loop is missing, the built-in thread coordinate variables are used to replace the array index, and there is no reference to N as it is implicitly defined by only launching N threads.
- Supposing a vector with the length of 32 elements, you can invoke the kernel with 32 threads as follows:
    ```c
        sumArraysOnGPU<<<1, 32>>>(d_A, d_B, d_C);
    ```