set(BINARY 02)
set(
    SOURCES 
    serial.h
    serial.cpp
    cuda.cuh
    cuda.cu
    main.cpp
)

add_executable(${BINARY} ${SOURCES})

# Link libMatrix
target_link_libraries(${BINARY} PRIVATE Matrix)

# Add 02 library
add_library(${BINARY}_lib STATIC ${SOURCES})

