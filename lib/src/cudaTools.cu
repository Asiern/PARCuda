#include "../include/cudaTools.cuh"
#include <stdlib.h>

size_t getDeviceMem(int device)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.totalGlobalMem;
}
