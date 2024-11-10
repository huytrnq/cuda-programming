# Sumation Example

This example is a simple example of how to use CUDA to sum an array of numbers. The example is taken from the book "Professional CUDA C Programming" by John Cheng, Max Grossman, and Ty McKercher.

## Run the Example
- Compile the code
    ```bash
    nvcc sumArraysonDevice.cu -o sumArraysonDevice
    ```
- Run the code
    ```bash
    ./sumArraysonDevice
    ```

## Code Explanation
- The code is divided into two files:
    - `sumArraysonDevice.cu` - This file contains the code to sum an array of numbers on the device.
- The code is divided into three functions:
    - `main` - This function initializes the array and calls the sum function.
    - `sumArraysonHost` - This function sums the array on the host.
    - `sumArraysonDevice` - This function sums the array on the device.

