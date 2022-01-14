#include "serial.h"
#include <stdlib.h>
#include <iostream>

int matrix_add(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    if (a != x || b != y)
        return 1;
#ifdef DEBUG
    //Define timer vars if DEBUG flag found
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    for (int i = 0; i < a * b; i++)
        out[i] = A[i] + B[i];
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    std::cout << "Ejec Add Serial (ms): " << std::fixed << time << std::endl;
#endif
    return 0;
}

int matrix_mul(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y)
{
    if (b != x)
        return 1;

#ifdef DEBUG
    //Define timer vars if DEBUG flag found
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < y; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < b; k++)
                sum = sum + A[i * b + k] * B[k * y + j];
            out[i * y + j] = sum;
        }
    }
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    std::cout << "Ejec Mul Serial (ms): " << std::fixed << time << std::endl;
#endif
    return 0;
}

int matrix_mul_add(float *A, float *B, float *C, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y, unsigned int p, unsigned int q)
{
    float *mul = (float *)malloc(sizeof(float) * a * y);
    if (matrix_mul(A, B, mul, a, b, x, y))
    {
        free(mul);
        return 1;
    }
    if (matrix_add(C, mul, out, p, q, a, y))
    {
        free(mul);
        return 1;
    }
    free(mul);
    return 0;
}