set(BINARY 05)
set(
    SOURCES
    main.cpp
    cuda.cu
    cuda.cuh
)

add_executable(${BINARY} ${SOURCES})

# Link libMatrix
target_link_libraries(${BINARY} PRIVATE Matrix)

# Link libCudaTools
target_link_libraries(${BINARY} PRIVATE CudaTools)

# Link 02
target_link_libraries(${BINARY} PRIVATE 02_lib)