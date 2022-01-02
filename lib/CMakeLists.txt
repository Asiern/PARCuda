# CudaTools
add_library(
    CudaTools
    STATIC
    src/cudaTools.cu
)

set_target_properties(CudaTools PROPERTIES PUBLIC_HEADER include/cudaTools.cuh)
target_include_directories(CudaTools PRIVATE include)
target_include_directories(CudaTools PRIVATE src)

# Matrix lib
add_library(
    Matrix
    STATIC
    src/matrix.cpp
)

set_target_properties(Matrix PROPERTIES PUBLIC_HEADER include/matrix.h)
target_include_directories(Matrix PRIVATE include)
target_include_directories(Matrix PRIVATE src)