# Define Grid and Block Example

This example is a simple example of how to define the grid and block size in CUDA. The example is taken from the book "Professional CUDA C Programming" by John Cheng, Max Grossman, and Ty McKercher.

## Run the Example
- Compile the code
    ```bash
    nvcc defineGridBlock.cu -o defineGridBlock
    ```
- Run the code
    ```bash
    ./defineGridBlock
    ```
- The output will be the grid and block size.

## Code Explanation
- The code is in a single file:
    - `defineGridBlock.cu` - This file contains the code to define the grid and block size.