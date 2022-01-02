#include "04.cuh"
#include "../03/cuda.cuh"
#include "matrix.h"
#include <stdlib.h>
#include <iostream>

int main(int argc, char const *argv[])
{
    int a = 5, b = 5, c = 5;
    if (argc == 1)
    {
        std::cout << "No se han pasado argumentos, se utilizarán los valores por defecto para el tamaño de las matrices." << std::endl;
        std::cout << "a=" << a << "; b=" << b << "; c=" << c << std::endl;
    }
    else if (argc == 4)
    {
        a = atoi(argv[1]);
        b = atoi(argv[2]);
        c = atoi(argv[3]);
    }
    else
    {
        std::cout << "Introduce las dimensiones de la matriz" << std::endl;
        std::cout << "./04 a b c" << std::endl;
        return 1;
    }

    // Reservar memoria para las matrices
    float *A = (float *)malloc(sizeof(float) * a * b);
    float *B = (float *)malloc(sizeof(float) * b * c);
    float *C = (float *)malloc(sizeof(float) * a * c);
    // Generar matrices
    gen_matrices(a, b, c, A, B, C);

    // Non shared
    std::cout << "\nEjecutando A * B + C con cuda" << std::endl;
    float *result = (float *)malloc(sizeof(float) * a * c);
    matrix_mul_add_cuda(A, B, C, result, a, b, b, c, a, c);
    print_matrix(result, a, c);

    // Shared
    std::cout << "\nEjecutando A * B + C con memoria compartida" << std::endl;
    float *result_shared = (float *)malloc(sizeof(float) * a * c);
    matrix_mul_add_cuda_shared(A, B, C, result_shared, a, b, b, c, a, c);
    print_matrix(result_shared, a, c);

    // Free
    free(A);
    free(B);
    free(C);
    free(result);
    free(result_shared);

    return 0;
}
