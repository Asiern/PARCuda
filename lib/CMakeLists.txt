add_library(
    Matrix
    STATIC
    src/matrix.cpp
)

set_target_properties(Matrix PROPERTIES PUBLIC_HEADER include/matrix.h)
target_include_directories(Matrix PRIVATE include)
target_include_directories(Matrix PRIVATE src)