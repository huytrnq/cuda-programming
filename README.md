# CUDA Programming

This repository is my personal study notes on CUDA programming. I will be using the book "Professonial CUDA C Programming" by John Cheng, Max Grossman, and Ty McKercher as the main reference. I will also be using the book "CUDA by Example" by Jason Sanders and Edward Kandrot as a secondary reference.

## Table of Contents
1. [Chapter 1: Hetergeneous Parallel Computing with CUDA](#chapter-1-hetergeneous-parallel-computing-with-cuda)
2. [Chapter 2: CUDA Programming Model](#chapter-2-cuda-programming-model)
3. [Chapter 3: CUDA Execution Model](#chapter-3-cuda-execution-model)
4. [Chapter 4: Global Memory](#chapter-4-global-memory)
5. [Chapter 5: Shared Memory and Constant Memory](#chapter-5-shared-memory-and-constant-memory)
6. [Chapter 6: Streams and Concurrency](#chapter-6-streams-and-concurrency)
7. [Chapter 7: Tuning Instruction-Level Primitive](#chapter-7-tuning-instruction-level-primitive)
8. [Chapter 8: GPU-Accelerated CUDA Libraries and OpenACC](#chapter-8-gpu-accelerated-cuda-libraries-and-openacc)
9. [Chapter 9 : Multi-GPU Programming](#chapter-9-multi-gpu-programming)
10. [Chapter 10: Implementation Considerations](#chapter-10-implementation-considerations)
11. [Important Notes](#important-notes)

## Chapter 1: Hetergeneous Parallel Computing with CUDA 
- [Introduction](./Chapter%201%20-%20Hetergeneous%20Parallel%20Computing%20with%20CUDA/Introduction.md)
- [Computer Architecture](./Chapter%201%20-%20Hetergeneous%20Parallel%20Computing%20with%20CUDA/ComputerArchitecture.md)
- [Parallel Computing](./Chapter%201%20-%20Hetergeneous%20Parallel%20Computing%20with%20CUDA/ParallelComputing.md)
- [Herterogeneous Computing](./Chapter%201%20-%20Hetergeneous%20Parallel%20Computing%20with%20CUDA/HeterogeneousComputing.md)
- [CUDA](./Chapter%201%20-%20Hetergeneous%20Parallel%20Computing%20with%20CUDA/CUDA.md)

## Chapter 2: CUDA Programming Model
- [Introduction](./Chapter%202%20-%20CUDA%20Programming%20Model/Introduction.md)
- [CUDA Programming Structure](./Chapter%202%20-%20CUDA%20Programming%20Model/CUDA%20Programing%20Structure.md)
- [Managing Memory](./Chapter%202%20-%20CUDA%20Programming%20Model/Managing%20Memory.md)
- [Organzing Threads](./Chapter%202%20-%20CUDA%20Programming%20Model/Organizing%20Threads.md)
- [Launching a CUDA Kernel](./Chapter%202%20-%20CUDA%20Programming%20Model/Launching%20a%20CUDA%20Kernel.md)
- [Writing Your Kernel](./Chapter%202%20-%20CUDA%20Programming%20Model/Writing%20Your%20Kernel.md)
- [Verifying Your Kernel](./Chapter%202%20-%20CUDA%20Programming%20Model/Verifying%20Your%20Kernel.md)
- [Handling Errors](./Chapter%202%20-%20CUDA%20Programming%20Model/Handling%20Error.md)
- [Timing with CPU Timer](./Chapter%202%20-%20CUDA%20Programming%20Model/Timing%20with%20CPU%20Timer.md)
- [Organizing Parallel Threads](./Chapter%202%20-%20CUDA%20Programming%20Model/Organizing%20Parallel%20Threads.md)
- [Managing Devices](./Chapter%202%20-%20CUDA%20Programming%20Model/Managing%20Devices.md)

## Chapter 3: CUDA Execution Model
- [Introduction](./Chapter%203%20-%20CUDA%20Execution%20Model/Introduction.md)
- [Understand the Nature of Warp Execution](./Chapter%203%20-%20CUDA%20Execution%20Model/Understand%20the%20Nature%20of%20Warp%20Execution.md)



## Chapter 4: Global Memory

## Chapter 5: Shared Memory and Constant Memory

## Chapter 6: Streams and Concurrency

## Chapter 7: Tuning Instruction-Level Primitive

## Chapter 8: GPU-Accelerated CUDA Libraries and OpenACC

## Chapter 9: Multi-GPU Programming

## Chapter 10: Implementation Considerations

## Important Notes
### Chapter 1: Hetergeneous Parallel Computing with CUDA
### Chapter 2: CUDA Programming Model
### Chapter 3: CUDA Execution Model
- Guidelines for Grid and Block Size: Using these guidelines will help your application scale on current and future devices:
    - Keep the number of threads per block a multiple of warp size (32). 
    - Avoid small block sizes: Start with at least 128 or 256 threads per block. 
    - Adjust block size up or down according to kernel resource requirements. 
    - Keep the number of blocks much greater than the number of SMs to expose sufficient parallelism to your device.
    - Conduct experiments to discover the best execution configuration and resource usage.