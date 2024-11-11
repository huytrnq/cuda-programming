# Managing Devices 

## Using the Runtime API to Query GPU Information
- You can use the following function to query all information about GPU devices:
    ```cpp
    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
    ```
- Depending on your configuration, checkDeviceInfor will report different information on the GPU device. 

## Determining the Best GPU 
- Some systems support multiple GPUs. In the case where each GPU is different, it may be important to select the best GPU to run your kernel. One way to identify the most computationally capable GPU is by the number of multiprocessors it contains. If you have a multi-GPU system, you can use the following code to select the most computationally capable device:
    ```cpp
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices > 1) {
        int maxMultiprocessors = 0, maxDevice = 0;
        for (int device=0; device<numDevices; device++) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            if (maxMultiprocessors < props.multiProcessorCount) {
                maxMultiprocessors = props.multiProcessorCount;
                maxDevice = device;
            }
        }
        cudaSetDevice(maxDevice);
    }
    ```

## Using nvidia-smi to Querry GPU Information
- The `nvidia-smi` command-line utility can be used to query information about the GPU devices on your system.
- You can use the following command to report details about GPU 0
    ```bash
        nvidia-smi -q -i 0
    ```

## Setting Devices at Runtime
- You can set the environment variable `CUDA_VISIBLE_DEVICES` to specify which GPU devices are visible to your application.
- For example, to set the first and third GPU devices as visible, you can use the following command:
    ```bash
    export CUDA_VISIBLE_DEVICES=0,2
    ```