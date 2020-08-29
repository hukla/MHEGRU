# cmake version to be used
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 14)

# project name
project(MHEGRU)

# flags

# files
file(GLOB MHEAAN_SOURCES "src/MHEAAN/*.cpp")
file(GLOB SOURCES "src/*.cpp")

# include
include_directories(./include ./src ./include/MHEAAN)

# target
add_executable(MHEGRU ${MHEAAN_SOURCES} ${SOURCES})

# external libs
#add_subdirectory(./libs/Logger)
#target_link_directories(binary logger)
target_link_libraries(MHEGRU -lpthread ntl m gmp)
