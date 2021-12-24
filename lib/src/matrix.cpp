#include "../include/matrix.h"
#include <random>
#include <iostream>

void generate(float *A, unsigned int n, unsigned int m)
{
    for (int i = 0; i < n * m; i++)
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Generate random numbers between 0 and 1
}

void print_matrix(float *A, unsigned int n, unsigned int m)
{
    std::cout << "(" << n << "x" << m << ")" << std::endl;
    std::cout.precision(2);
    for (int i = 0; i < n * m; i++)
    {
        std::cout << A[i] << "\t";
        if ((i + 1) % m == 0)
            std::cout << std::endl;
    }
}

void gen_matrices(unsigned int a, unsigned int b, unsigned int c, float *A, float *B, float *C)
{
    // Generate A
    generate(A, a, b);

    // Generate B
    generate(B, b, c);

    // Generate C
    generate(C, a, c);
}