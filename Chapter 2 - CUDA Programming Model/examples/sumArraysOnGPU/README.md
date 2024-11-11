# Summing Arrays on the GPU with different configurations

This example demonstrates how to sum two arrays on the GPU using different configurations. The configurations are as follows:
- 2D Grid and 2D Block
- 1D Grid and 1D Block
- 2D Grid and 1D Block

## Folder Structure
- [sumArraysOnGPU-small-case.cu](./sumArraysOnGPU-small-case.cu): This file contains the code for summing two small arrays on the GPU on Host.
- [sumArraysOnGPU-2D-Grid-2D-Block.cu](./sumArraysOnGPU-2D-Grid-2D-Block.cu): This file contains the code for summing two arrays on the GPU using 2D Grid and 2D Block.
- [sumArraysOnGPU-1D-Grid-1D-Block.cu](./sumArraysOnGPU-1D-Grid-1D-Block.cu): This file contains the code for summing two arrays on the GPU using 1D Grid and 1D Block.
- [sumArraysOnGPU-2D-Grid-1D-Block.cu](./sumArraysOnGPU-2D-Grid-1D-Block.cu): This file contains the code for summing two arrays on the GPU using 2D Grid and 1D Block.

## Running the code
To run the code, use the following command:
```bash
nvcc sumArraysOnGPU-2D-Grid-2D-Block.cu -o sumArraysOnGPU-2D-Grid-2D-Block
./sumArraysOnGPU-2D-Grid-2D-Block
```
