# Check Dimension Example

This example is a simple example of how to check the dimension of the CUDA device. The example is taken from the book "Professional CUDA C Programming" by John Cheng, Max Grossman, and Ty McKercher.

## Run the Example
- Compile the code
    ```bash
    nvcc checkDimension.cu -o checkDimension
    ```
- Run the code
    ```bash
    ./checkDimension
    ```
- The output will be the dimension of the CUDA device.

## Code Explanation
- The code is divided into two files:
    - `checkDimension.cu` - This file contains the code to check the dimension of the CUDA device.
- The code is divided into two functions:
    - `main` - This function initializes the CUDA device and calls the checkDimension function.
    - `checkDimension` - This function checks the dimension of the CUDA device.
