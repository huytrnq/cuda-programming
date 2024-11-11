# Managing Devices 

## Using the Runtime API to Query GPU Information
- You can use the following function to query all information about GPU devices:
    ```cpp
    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
    ```
- Depending on your configuration, checkDeviceInfor will report different information on the GPU device. 