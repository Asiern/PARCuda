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

target_link_libraries(${BINARY} PRIVATE Matrix)

