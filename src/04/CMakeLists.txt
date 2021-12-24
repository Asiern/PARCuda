set(BINARY 04)
set(
    SOURCES 
    04.cuh
    04.cu
    main.cpp
)

add_executable(${BINARY} ${SOURCES})

target_link_libraries(${BINARY} PRIVATE Matrix)
