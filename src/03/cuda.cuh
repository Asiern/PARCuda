int matrix_add_cuda(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y);
int matrix_mul_cuda(float *A, float *B, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y);
int matrix_mul_add_cuda(float *A, float *B, float *C, float *out, unsigned int a, unsigned int b, unsigned int x, unsigned int y, unsigned int p, unsigned int q);