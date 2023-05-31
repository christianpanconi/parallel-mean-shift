# mean_shift_module
cmake_minimum_required(VERSION 3.18)
project(mean_shift_module CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE) # default built type to Release
	set(CMAKE_BUILD_TYPE Release)
endif()
message( STATUS "Build type: ${CMAKE_BUILD_TYPE}" )

message(STATUS "Configured CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

# Setup CUDA
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Setup python and numpy
find_package(Python REQUIRED Development NumPy)

# Module target
add_subdirectory( ../lib/mean_shift ./lib )
set( SOURCE_DIR "./mean_shift" )
add_library(mean_shift_module SHARED 
	${SOURCE_DIR}/timer.hpp
	${SOURCE_DIR}/mean_shift_py_module_utils.h
	${SOURCE_DIR}/mean_shift_py_wrappers.h
	${SOURCE_DIR}/mean_shift_py_wrappers.cpp
	${SOURCE_DIR}/cumean_shift_py_wrappers.h 
	${SOURCE_DIR}/cumean_shift_py_wrappers.cu
	${SOURCE_DIR}/mean_shift_py_module.h  
	${SOURCE_DIR}/mean_shift_py_module.cpp )
target_link_libraries(mean_shift_module
	PRIVATE mean_shift 
	PRIVATE Python::Python 
	PRIVATE Python::NumPy )
#set_property( TARGET mean_shift_module PROPERTY CUDA_SEPARABLE_COMPILATION ON)	 
	
