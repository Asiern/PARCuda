#include "04.cuh"
#include <iostream>

#define TILE_DIM 16
#define n_threads 1024

__global__ void add_kernel_shared(float *A, float *B, float *out, unsigned int n, unsigned int m)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    out[x] = A[x] + B[x];
}

__global__ void mul_kernel_shared(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    // Shared variables
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];

    // Indexes
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0;

    for (int i = 0; i < (b - 1) / TILE_DIM + 1; i++)
    {
        if (row < a && i * TILE_DIM + threadIdx.x < b)
            A_shared[threadIdx.y][threadIdx.x] = A[row * b + i * x + col];
        else
            A_shared[threadIdx.y][threadIdx.x] = 0;
        if (col < y && i * TILE_DIM + threadIdx.y < x)
            B_shared[threadIdx.y][threadIdx.x] = B[(i * TILE_DIM + threadIdx.y) * y + col];
        else
            B_shared[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE_DIM; k++)
        sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
    __syncthreads();

    if (row < a && col < y)
        out[row * y + col] = sum;
}

int matrix_add_cuda_shared(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
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
    {
        std::cout << "Error no se puede reservar memoria (Mat add)" << std::endl;
        return 1;
    }

    float *d_B;
    cudaMalloc(&d_B, size);
    if (cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        std::cout << "Error al copiar matriz a memoria (Mat add)" << std::endl;
        return 1;
    }

    float *d_out;
    if (cudaMalloc(&d_out, size) != cudaSuccess)
    {
        std::cout << "Error no se puede reservar memoria (Mat add)" << std::endl;
        return 1;
    }

    // Call Kernel
#ifdef DEBUG
    cudaEventRecord(start);
#endif
    // TODO threads
    add_kernel_shared<<<32, n_threads>>>(d_A, d_B, d_out, a, b);
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

int matrix_mul_cuda_shared(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    // Return if matrix dimensions not compatible
    if (b != x)
        return 1;

#ifdef DEBUG
    // Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // Allocate Memory

    // A Matrix
    float *d_A;
    size_t sizeA = sizeof(float) * a * b;
    if (cudaMalloc(&d_A, sizeA) == cudaErrorMemoryAllocation)
    {
        std::cout << "Error no se puede reservar memoria" << std::endl;
        return 1;
    }
    if (cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice) > 0)
    {
        std::cout << "Error al copiar matriz a memoria" << std::endl;
        return 1;
    }

    // B Matrix
    float *d_B;
    size_t sizeB = sizeof(float) * x * y;
    if (cudaMalloc(&d_B, sizeB) == cudaErrorMemoryAllocation)
    {
        std::cout << "Error no se puede reservar memoria" << std::endl;
        return 1;
    }
    if (cudaMemcpy(d_B, B, sizeA, cudaMemcpyHostToDevice) > 0)
    {
        std::cout << "Error al copiar matriz a memoria" << std::endl;
        return 1;
    }

    // Out Matrix
    float *d_out;
    size_t sizeOut = sizeof(float) * a * y;
    if (cudaMalloc(&d_out, sizeOut) == cudaErrorMemoryAllocation)
    {
        std::cout << "Error no se puede reservar memoria" << std::endl;
        return 1;
    }

    dim3 dimGrid((y / TILE_DIM) + 1, (a / TILE_DIM) + 1, 1); //Number of Blocks required
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);                    //Number of threads in each block

#ifdef DEBUG
    // Start timer
    cudaEventRecord(start);
#endif
    mul_kernel_shared<<<dimGrid, dimBlock>>>(d_A, d_B, d_out, a, b, x, y);
    cudaDeviceSynchronize();
#ifdef DEBUG
    // TODO fix timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Cuda shared mul kernel ex time(ms): " << time_ms << std::endl;
#endif

    // Copy results
    cudaMemcpy(out, d_out, sizeOut, cudaMemcpyDeviceToHost);

    // Free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    return 0;
}

int matrix_mul_add_cuda_shared(float *A, float *B, float *C, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y, unsigned int p, unsigned int q)
{
    float *mul = (float *)malloc(sizeof(float) * a * y);
    if (matrix_mul_cuda_shared(A, B, mul, a, b, x, y))
    {
        free(mul);
        return 1;
    }
    if (matrix_add_cuda_shared(C, mul, out, p, q, a, y))
    {
        free(mul);
        return 1;
    }
    free(mul);
    return 0;
}