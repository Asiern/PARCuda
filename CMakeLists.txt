cmake_minimum_required(VERSION 3.12)
project(PAR_Cuda LANGUAGES CXX CUDA)

# Add debug option
option(USE_DEBUG "Debug mode" OFF)
if(USE_DEBUG)
    message("Using DEBUG definition")
    add_definitions(-DDEBUG=1)
endif()

# Include libMatrix
include_directories(lib/include)
add_subdirectory(lib)

# Source dirs
add_subdirectory(src/01)
add_subdirectory(src/02)
add_subdirectory(src/03)
add_subdirectory(src/04)
add_subdirectory(src/05)