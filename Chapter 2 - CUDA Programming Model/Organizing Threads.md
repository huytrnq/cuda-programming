# Organizing Threads

- CUDA exposes a two-level abstraction hierarchy for organizing threads: **grids** of blocks and **blocks** of threads to enable you to organize your threads.
    ![Threads Hierarchy](./images/ThreadsHierarchy.png)

- All threads spawned by a single kernel launch are collectively called a grid. All threads in a grid share the same global memory space. A grid is made up of many thread blocks. A thread block is a group of threads that can cooperate with each other using:
    - Block-local synchronization
    - Block-local shared memory
- Threads from different blocks cannot cooperate.
- Threads rely on the following two unique coordinates to distinguish themselves from each other:
    - blockIdx (block index within a grid)
    - threadIdx (thread index within a block)
- These variables appear as built-in, pre-initialized variables that can be accessed within kernel functions. When a kernel function is executed, the coordinate variables blockIdx and threadIdx are assigned to each thread by the CUDA runtime. Based on the coordinates, you can assign portions of data to different threads.
- The coordinat variables are of type uint3, which is a CUDA-specific data type that is a three-dimensional vector. The x, y, and z components of the dim3 type are used to represent the x, y, and z dimensions of the grid and block: blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, and threadIdx.z.
- The dimensions of a grid and block are specified by the following two built-in variables:
    - gridDim  (grid dimension, measured in blocks)
    - blockDim (block dimension, measured in threads)
- Usually a grid is organized as a two-dimensional array of blocks, and a block is organized as a three-dimensional array of threads.