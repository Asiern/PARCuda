#include "../include/cudaTools.cuh"

size_t getDeviceMem(int device)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.totalGlobalMem;
}
