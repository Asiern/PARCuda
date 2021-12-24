#include "cuda.cuh"
#include <iostream>

#define n_threads 1024

__global__ void add_kernel(float *A, float *B, float *out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    out[x] = A[x] + B[x];
}
__global__ void mul_kernel(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    int row = blockIdx.y * a + threadIdx.y;
    int col = blockIdx.x * a + threadIdx.x;

    if (row < a && col < a)
    {
        float sum = 0;
        for (int i = 0; i < a; i++)
            sum += A[row * a + i] * B[i * a + col];
        out[row * a + col] = sum;
    }
}

int matrix_add_cuda(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    if (a != x || b != y)
        return 1;

#ifdef DEBUG
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    size_t size = sizeof(float) * a * b;

    // Allocate Memory
    float *d_A;
    cudaMalloc(&d_A, size);
    if (cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice) != cudaSuccess)
        return 1;

    float *d_B;
    cudaMalloc(&d_B, size);
    if (cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice) != cudaSuccess)
        return 1;

    float *d_out;
    cudaMalloc(&d_out, size);

    // Call Kernel
#ifdef DEBUG
    cudaEventRecord(start);
#endif
    add_kernel<<<32, n_threads>>>(d_A, d_B, d_out);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Cuda add kernel ex time(ms): " << time_ms << std::endl;
#endif

    // Copy results
    if (cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost) != cudaSuccess)
        return 1;

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
    return 0;
}

int matrix_mul_cuda(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    if (b != x)
        return 1;

#ifdef DEBUG
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // Allocate Memory
    float *d_A;
    size_t sizeA = sizeof(float) * a * b;
    cudaMalloc(&d_A, sizeA);
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    float *d_B;
    size_t sizeB = sizeof(float) * x * y;
    cudaMalloc(&d_B, sizeB);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    float *d_out;
    size_t sizeOut = sizeof(float) * a * y;
    cudaMalloc(&d_out, sizeOut);

    dim3 n_blocks = dim3(a * b / n_threads);

#ifdef DEBUG
    cudaEventRecord(start);
#endif
    // Call kernel
    mul_kernel<<<n_blocks, n_threads>>>(d_A, d_B, d_out, a, b, x, y);
    cudaDeviceSynchronize();
#ifdef DEBUG
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Cuda mul kernel ex time(ms): " << time_ms << std::endl;
#endif

    // Copy results
    cudaMemcpy(out, d_out, sizeOut, cudaMemcpyDeviceToHost);

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    return 0;
}

int matrix_mul_add_cuda(float *A, float *B, float *C, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y, unsigned int p, unsigned int q)
{
    float *mul = (float *)malloc(sizeof(float) * a * y);
    if (matrix_mul_cuda(A, B, mul, a, b, x, y))
    {
        free(mul);
        return 1;
    }
    if (matrix_add_cuda(C, mul, out, p, q, a, y))
    {
        free(mul);
        return 1;
    }
    free(mul);
    return 0;
}
