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
        std::cout << "No se han pasado argumentos, se utilizarán los valores por defecto para el tamaño de las matrices." << std::endl;
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
    struct timespec t1, t2;
#endif

    size_t size = n * m;
    float *A = (float *)malloc(sizeof(float) * size);
    float *B = (float *)malloc(sizeof(float) * size);

    // Generar matriz A
    generate(A, n, m);
    // Copiar matriz a en B
    std::memcpy(B, A, size * sizeof(float));

    std::cout << "\nMatriz generada A" << std::endl;
    print_matrix(A, n, m);
    std::cout << "\nMatriz generada B" << std::endl;
    print_matrix(B, n, m);

    // Serial
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &t1);
#endif
    transpose(A, n, m);
#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &t2);
    printf("Ejec Transpose Serial (ns): %f\n", round((t2.tv_nsec - t1.tv_nsec) / 1.0e6));
#endif
    std::cout << "Imprimiendo resultado de la matriz Serie" << std::endl;
    print_matrix(A, m, n);

    // Cuda
    transpose_cuda(B, n, m);
    std::cout << "Imprimiendo resultado de la matriz CUDA" << std::endl;
    print_matrix(B, m, n);

    // Free
    free(A);
    free(B);

    return 0;
}
