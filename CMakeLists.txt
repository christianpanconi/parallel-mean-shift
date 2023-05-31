cmake_minimum_required(VERSION 3.18)
project(mean_shift CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory( lib/mean_shift ./lib )