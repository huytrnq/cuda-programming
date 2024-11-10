# Verifying Your Kernel

- Now that you have written your kernel, you need to verify that it is working correctly. You need a host function to verify the result from the kernel.
```c
    void checkResult(float *hostRef, float *gpuRef, const int N) {
        double epsilon = 1.0E-8;
        int match = 1;
        for (int i = 0; i < N; i++) {
            if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
                match = 0;
                printf("Arrays do not match!\n");
                printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
                break;
            }
        }
        if (match) printf("Arrays match.\n\n");
        return;
    }
```

### Verifying Kernel Code
- Besides many useful debugging tools, there are two very basic but useful means by which you can verify your kernel code. 
    - First, you can use printf in your kernel for Fermi and later generation devices.
    - Second, you can set the execution confi guration to <<<1,1>>>, so you force the kernel to run with only one block and one thread. This emulates a sequential implementation. This is useful for debugging and verifying correct results. Also, this helps you verify that numeric results are bitwise exact from run-to-run if you encounter order of operations issues.