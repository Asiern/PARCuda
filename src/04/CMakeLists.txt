set(BINARY 04)
set(
    SOURCES 
    04.cuh
    04.cu
    main.cpp
)

add_executable(${BINARY} ${SOURCES})

# Link libMatrix
target_link_libraries(${BINARY} PRIVATE Matrix)

# Add 03_lib
target_link_libraries(${BINARY} PRIVATE 03_lib)