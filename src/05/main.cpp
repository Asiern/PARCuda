#include "cuda.cuh"
#include <stdlib.h>
#include "matrix.h"
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
        std::cout << "./05 a b c" << std::endl;
        return 1;
    }

    // Reservar memoria para las matrices
    float *A = (float *)malloc(sizeof(float) * a * b);
    float *B = (float *)malloc(sizeof(float) * b * c);
    float *C = (float *)malloc(sizeof(float) * a * c);

    // Generar matrices
    gen_matrices(a, b, c, A, B, C);

    float *result = (float *)malloc(sizeof(float) * a * b);
    matrix_add_cuda_big(A, A, result, a, b, a, b);
    // if (matrix_mul_cuda_big(A, B, out, n, n, n, n) > 0)
    // {
    //     return 1;
    // }

    // print_matrix(result, a, c);
    return 0;
}
