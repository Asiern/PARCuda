#include "cuda.cuh"
#include "cudaTools.cuh"
#include "matrix.h"
#include "../02/serial.h"
#include <stdlib.h>
#include <cstring>
#include <stdio.h>
#include <iostream>
#include <algorithm> // min

#define NTHREADS 1024

__global__ void add_kernel(float *A, float *B, float *out, size_t nElem)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > nElem)
        return;
    out[x] = A[x] + B[x];
}

int matrix_add_cuda_big(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    if (a != x || b != y)
        return 1;

#ifdef DEBUG
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // Calculate max number of rows to copy
    size_t deviceMemSize = getDeviceMem(0);    // Device mem size in bytes
    size_t matrixSize = sizeof(float) * a * b; // Matrix size in bytes

    int maxElems = (deviceMemSize / sizeof(float)) - 1;
    int elemsPerMatrix = maxElems / 3; // We have 3 matrix

    elemsPerMatrix = std::min(elemsPerMatrix, (int)matrixSize);

    // Allocate device mem
    float *d_A, *d_B, *d_out;
    if (cudaMalloc(&d_A, elemsPerMatrix * sizeof(float)) != cudaSuccess)
    {
        printf("Error allocate mem\n");
        return 1;
    }
    if (cudaMalloc(&d_B, elemsPerMatrix * sizeof(float)) != cudaSuccess)
    {
        printf("Error allocate mem\n");
        return 1;
    }
    if (cudaMalloc(&d_out, elemsPerMatrix * sizeof(float)) != cudaSuccess)
    {
        printf("Error allocate mem\n");
        return 1;
    }
#ifdef DEBUG
    cudaEventRecord(start);
#endif
    // Loop until all elements are processed
    for (int i = 0; i < a * b; i += elemsPerMatrix)
    {
        if (cudaMemcpy(d_A, A + i, elemsPerMatrix * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            printf("Error al copiar mem A\n");
            return 1;
        }
        if (cudaMemcpy(d_B, B + i, elemsPerMatrix * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            printf("Error al copiar mem B\n");
            return 1;
        }

        // Calculate number of blocks to lauch
        dim3 nblocks((elemsPerMatrix / NTHREADS) + 1);

        // Launch kernel
        add_kernel<<<nblocks, NTHREADS>>>(d_A, d_B, d_out, elemsPerMatrix);

        cudaDeviceSynchronize();

        // Save results
        if (cudaMemcpy(&(out[i]), d_out, elemsPerMatrix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            printf("Error al copiar mem OUT");
            printf("index %d\n", i);
            return 1;
        }
    }

#ifdef DEBUG
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Cuda add big kernel ex time(ms): " << time_ms << std::endl;
#endif

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    return 0;
}

__global__ void mul_kernel(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int nElem)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x > nElem)
        return;
    *out += A[x] * B[x];
}

int matrix_mul_cuda_big(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    if (b != x)
        return 1;

    // Transpose B matrix to improve mem locality
    size_t sizeB = sizeof(float) * x * y;
    float *Btrans = (float *)malloc(sizeB);
    std::memcpy(Btrans, B, sizeB);
    // TODO use cuda to transpose?
    transpose(Btrans, x, y);

    // Calculate max number of rows to copy
    size_t deviceMemSize = getDeviceMem(0);                        // Device mem size in bytes
    size_t maxElemsOnDevice = (deviceMemSize / sizeof(float)) - 1; // Max number of floats on card mem
    size_t maxRowSize = (maxElemsOnDevice - 1) / 2;                // Max Row size in bytes
    size_t rowSize = sizeof(float) * b;                            // Row size in bytes
    size_t rowSlice = min(rowSize, maxRowSize);                    // Row slice size in bytes
    // size_t numOfPieces = ceil(rowSize / maxRowSize);               // Number of times a row has te be copied

    float *d_A, *d_B, *d_out, *result;
    result = (float *)malloc(sizeof(float));
    if (cudaMalloc(&d_A, rowSlice) != cudaSuccess)
    {
        printf("Could not allocate mem on device\n");
        return 1;
    }
    if (cudaMalloc(&d_B, rowSlice) != cudaSuccess)
    {
        printf("Could not allocate mem on device\n");
        return 1;
    }
    if (cudaMalloc(&d_out, sizeof(float)) != cudaSuccess)
    {
        printf("Could not allocate mem on device\n");
        return 1;
    }

    // For each row in a
    for (int i = 0; i < a; i++)
    {
        // For each row in BTrans
        for (int j = 0; j < y; j++)
        {
        }
    }

    // Free
    free(Btrans);
    free(result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    return 0;
}

int matrix_mul_add_cuda_big(float *A, float *B, float *C, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y, unsigned int p, unsigned int q)
{
    float *mul = (float *)malloc(sizeof(float) * a * y);
    if (matrix_mul_cuda_big(A, B, mul, a, b, x, y))
    {
        free(mul);
        return 1;
    }
    if (matrix_add_cuda_big(C, mul, out, p, q, a, y))
    {
        free(mul);
        return 1;
    }
    free(mul);
    return 0;
}