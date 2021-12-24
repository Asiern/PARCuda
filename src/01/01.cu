#include "01.cuh"
#include <iostream>

void printDevProp(cudaDeviceProp devProp)
{
    std::cout << "Nombre: " << devProp.name << std ::endl;
    std::cout << "Memoria total: " << devProp.totalGlobalMem / 1073741824.0f << " GB" << std ::endl;
    std::cout << "Memoria compartida por bloque: " << devProp.sharedMemPerBlock << " bytes" << std ::endl;
    std::cout << "Tamaño del warp: " << devProp.warpSize << std ::endl;
    std::cout << "Número de multiprocesadores: " << devProp.multiProcessorCount << std ::endl;
    std::cout << "Frecuencia del reloj: " << devProp.clockRate << " KHz" << std ::endl;
    std::cout << "ECC: " << devProp.ECCEnabled << std::endl;
    std::cout << "Tamaño de la cache L2: " << devProp.l2CacheSize << " bytes" << std::endl;
}

void show_properties(void)
{
    int i, devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << devCount << " CUDA Devices" << std::endl;
    for (i = 0; i < devCount; i++)
    {
        std::cout << "CUDA Device #" << i << std::endl;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
}