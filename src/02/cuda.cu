#include "cuda.cuh"
#include <iostream>

__global__ void transpose_kernel(float *in, float *out, int n, int m)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int fila = x % m;
    unsigned int columna = x / m;
    unsigned int y = fila * n + columna;
    out[x] = in[y];
}

void transpose_cuda(float *C, unsigned int N, unsigned int M)
{

#ifdef DEBUG
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // Create memory copies
    size_t size = sizeof(float) * N * M;
    float *in, *out;
    cudaMalloc(&in, size);
    cudaMalloc(&out, size);
    cudaMemcpy(in, C, size, cudaMemcpyHostToDevice);

#ifdef DEBUG
    cudaEventRecord(start);
#endif
    // Launch Kernel
    transpose_kernel<<<32, 32>>>(in, out, N, M);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Cuda mul kernel ex time(ns): " << time_ms * 1000000 << std::endl;
#endif

    // Save results
    cudaMemcpy(C, out, size, cudaMemcpyDeviceToHost);

    // Free
    cudaFree(in);
    cudaFree(out);
}
