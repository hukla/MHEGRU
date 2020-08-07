cmake_minimum_required(VERSION 3.10)
project(MHEGRU)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(MHEGRU
#        src/AddingProblem.cpp src/AddingProblem.h
        src/MHEAddingProblem.cpp src/MHEAddingProblem.h
        MHEAAN/)
