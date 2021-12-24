set(BINARY 05)
set(
    SOURCES
    main.cpp
)

add_executable(${BINARY} ${SOURCES})

target_link_libraries(${BINARY} PRIVATE Matrix)
