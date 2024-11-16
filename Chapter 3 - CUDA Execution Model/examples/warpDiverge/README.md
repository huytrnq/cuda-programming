# Warp Divergence

In this example, we will demonstrate the concept of warp divergence. Warp divergence occurs when threads in the same warp take different paths through the code. This can lead to a decrease in performance because the warp executes serially each branch path, disabling threads that do not take this path.

## Using nvight-systems to check for warp divergence
```bash
ncu --metrics smsp__sass_branch_targets_threads_divergent.avg,smsp__sass_branch_targets_threads_divergent.sum ./test
```

## Simple Divergence Example
- simpleDivergence.cu: contains three kernels that demonstrate warp divergence. 
    - `mathKernel1` sets the value of `a` to 100.0f if the thread ID is even, and sets the value of `b` to 200.0f if the thread ID is odd.
    - `mathKernel2` sets the value of `a` to 100.0f if the thread ID is even, and sets the value of `b` to 200.0f if the thread ID is odd. This kernel is designed to avoid warp divergence.
    - `mathKernel3` similar to `mathKernel1` but cause more warp divergence as the number of conditional branches increases 
