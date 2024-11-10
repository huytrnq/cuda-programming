# Handling Errors

- Since many CUDA calls are asynchronous, it may be diffi cult to identify which routine caused an error. Defining an error-handling macro to wrap all CUDA API calls simplifi es the error checking process:
```cpp
    #define CHECK(call) 
    { 
        const cudaError_t error = call; 
        if (error != cudaSuccess) 
        { 
            printf("Error: %s:%d, ", __FILE__, __LINE__); 
            printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); 
            exit(1); 
        } 
    }
```
- This macro can be used to check the return value of any CUDA API call. For example, the following code snippet checks the return value of cudaMalloc:
```cpp
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
```
- It can be also used after a kernel invocation to check for kernel errors:
```cpp
    kernel_function<<<grid, block>>>(argument list);
    CHECK(cudaDeviceSynchronize());
``` 
- CHECK(cudaDeviceSynchronize()) blocks the host thread until the kernel has completed all the preceding requested tasks, and ensures that no errors occurred as part of the last kernel launch. **This technique should be used just for debugging purposes, because adding this check point after kernel launches will block the host thread and make that point a global barrier.**