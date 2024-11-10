# Timer

This example demonstrates how to use a CPU timer to measure the time taken by a kernel. The CPU timer is created using the `gettimeofday()` function. The `gettimeofday()` function is defined in the `sys/time.h` header file. The `cpuSecond()` function returns the current time in seconds. The time taken by the kernel is measured by calling the `cpuSecond()` function before and after the kernel execution. The difference between the two times gives the time taken by the kernel.

- Compile the code
```bash
    nvcc -o sumArraysOnGPU-timer sumArraysOnGPU-timer.cu
```

- Run the code
```bash
    ./sumArraysOnGPU-timer
```

- Use the `nvprof` tool to measure the time taken by the kernel
```bash
    nvprof ./sumArraysOnGPU-timer
```
