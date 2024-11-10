# Timing with CPU Timer

- A CPU timer can be created by using the gettimeofday() function. You need to include the sys/time.h header file to use this function.
```c
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
    }
```
- You can measure your kernel with cpuSecond() function as follows:
```c
    double iStart = cpuSecond();
    kernel_name<<<grid, block>>>(argument list);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    double iElaps = cpuSecond() - iStart;
```

## Timing with nvprof
- You can also use the nvprof tool to measure the time taken by your kernel. You can use the following command to measure the time taken by your kernel:
```bash
nvprof ./executable
```
- The output will show the time taken by the kernel in milliseconds.
    ```bash
    ./sumArraysOnGPU-timer Starting...
    ==10642== NVPROF is profiling process 10642, command: ./sumArraysOnGPU-timer
    Vector size 16777216
    sumArraysOnGPU <<<16384, 1024>>> Time elapsed 0.001134 sec
    sumArraysOnHost Time elapsed 0.051119 sec
    Arrays match.

    ==10642== Profiling application: ./sumArraysOnGPU-timer
    ==10642== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    GPU activities:   64.48%  27.909ms         2  13.955ms  13.914ms  13.995ms  [CUDA memcpy HtoD]
                    33.48%  14.493ms         1  14.493ms  14.493ms  14.493ms  [CUDA memcpy DtoH]
                        2.03%  878.96us         1  878.96us  878.96us  878.96us  sumArraysOnGPU(float*, float*, float*, int)
        API calls:   54.74%  93.806ms         1  93.806ms  93.806ms  93.806ms  cudaSetDevice
                    25.22%  43.217ms         3  14.406ms  14.137ms  14.874ms  cudaMemcpy
                    16.84%  28.863ms         1  28.863ms  28.863ms  28.863ms  cudaDeviceReset
                        2.06%  3.5339ms         3  1.1780ms  297.12us  2.1363ms  cudaFree
                        0.51%  882.15us         1  882.15us  882.15us  882.15us  cudaDeviceSynchronize
                        0.39%  660.00us         3  220.00us  137.91us  364.21us  cudaMalloc
                        0.14%  246.15us         1  246.15us  246.15us  246.15us  cudaLaunchKernel
                        0.08%  139.94us       114  1.2270us     134ns  55.764us  cuDeviceGetAttribute
                        0.01%  14.274us         1  14.274us  14.274us  14.274us  cuDeviceGetName
                        0.00%  5.2380us         1  5.2380us  5.2380us  5.2380us  cuDeviceGetPCIBusId
                        0.00%  4.8250us         1  4.8250us  4.8250us  4.8250us  cuDeviceTotalMem
                        0.00%  1.5660us         3     522ns     147ns  1.0240us  cuDeviceGetCount
                        0.00%  1.1730us         2     586ns     311ns     862ns  cuDeviceGet
                        0.00%     430ns         1     430ns     430ns     430ns  cuModuleGetLoadingMode
                        0.00%     262ns         1     262ns     262ns     262ns  cuDeviceGetUuid
    ```

- For HPC workloads, it is important to understand the compute to communication ratio. If your application spends more time computing than transferring data, then it may be possible to overlap these operations and completely hide the latency associated with transferring
data. If your application spends less time computing than transferring data, it is important to minimize the transfer between the host and device. 

## Comparing App performance to Maximum Theoretical Performance
- While performing application optimization, it is important to compare the performance of your application to the maximum theoretical performance. The maximum theoretical performance is the peak performance of the hardware. For example:
    - Tesla K10 Peak Single Precision Performance: 745 MHz core clock * 2 GPUs/board * (8 multiprocessors * 192 fp32 cores/multiprocessor) * 2 ops/cycle = 4.58 TFLOPS
    - Tesla K10 Peak Memory Bandwidth: 2 GPUs/board * 256 bit * 2500 MHz mem-clock * 2 DDR / 8 bits/byte = 320 GB/s
    - Ratio of instruction:bytes: 4.58 TFLOPS / 320 GB/s yields 13.6 instructions:1 byte
- If your application issues more than 13.6 instructions for every byte accessed, then your application is bound by arithmetic performance. Most HPC workloads are bound by memory bandwidth.