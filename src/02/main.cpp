#include "cuda.cuh"
#include "serial.h"
#include "matrix.h"

#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <math.h>

#ifdef DEBUG
#include <time.h>
#endif

int main(int argc, char *argv[])
{
    int n = 5, m = 5;
    if (argc == 1)
        std::cout << "No se han pasado argumentos, se utilizarán los valores por defecto para el tamaño de las matrices. (5x5)" << std::endl;
    else if (argc == 3)
    {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
    }
    else
    {
        std::cout << "Introduce las dimensiones de la matriz" << std::endl;
        std::cout << "./02 n m" << std::endl;
        return 1;
    }

#ifdef DEBUG
    //Define timer vars if DEBUG flag found
    struct timespec start, end;
#endif

    size_t size = n * m;
    float *A = (float *)malloc(sizeof(float) * size);
    float *B = (float *)malloc(sizeof(float) * size);

    // Generar matriz A
    generate(A, n, m);
    // Copiar matriz A en B
    std::memcpy(B, A, size * sizeof(float));
    // print_matrix(B, n, m);

    std::cout << "\nMatriz A generada" << std::endl;
    std::cout << "Matriz B generada\n"
              << std::endl;

    // Serial
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    transpose(A, n, m);
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    std::cout << "Ejec Transpose Serial (ms): " << std::fixed << time << std::endl;
#endif

    // Cuda
    transpose_cuda(B, n, m);
    // print_matrix(B, n, m);

    // Free
    free(A);
    free(B);

    return 0;
}
