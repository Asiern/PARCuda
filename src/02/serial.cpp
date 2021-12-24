#include <stdlib.h>
#include <cstdint>
#include <cstring>

void transpose(float *A, unsigned int N, unsigned int M)
{
    size_t size = N * M;
    float *tmpM = (float *)malloc(sizeof(float) * size);

    // Copy C => tmpM
    std::memcpy(tmpM, A, size * sizeof(float));

    // Transpose matrix
    int k = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            A[k++] = tmpM[j * N + i];

    // Free
    free(tmpM);
}