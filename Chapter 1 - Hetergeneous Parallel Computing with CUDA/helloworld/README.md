# Hello World Example

This is the hello world example for CUDA programming. The code is simple and is used to demonstrate the basic structure of a CUDA program.

## Compilation
To compile the code, use the following command:
```bash
nvcc hello.cu -o hello
```

## Running the Code
To run the code, use the following command:
```bash
./hello
```

## Output
The output of the code should be:
```
Hello World from CPU
Hello World from GPU
Hello World from GPU
Hello World from GPU
Hello World from GPU
Hello World from GPU
Hello World from GPU
Hello World from GPU
Hello World from GPU
Hello World from GPU
Hello World from GPU
```

## Note 
- cudaDeviceReset() 
    - Purpose: Resets the device, releasing all resources (memory allocations, kernel execution, etc.) associated with the current CUDA context.
    - Usage: Typically called at the end of a CUDA program to clean up resources. This function also implicitly calls cudaDeviceSynchronize.
    - Typical Scenario: Use this when you are completely done with all CUDA operations and want to ensure a clean slate, such as when exiting a program or resetting the device after a major error.
- cudaDeviceSynchronize() 
    - Purpose: Ensures that all preceding CUDA tasks are completed before proceeding to the next CPU code.
    - Usage: Called when you want to make sure all previous kernel executions, memory operations, and other asynchronous operations have finished.
    - Typical Scenario: Use this if you want to wait until the device has completed all tasks, especially useful for debugging to detect errors in asynchronous operations.
