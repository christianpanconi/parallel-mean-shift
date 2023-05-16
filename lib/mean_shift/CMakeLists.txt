# mean_shift_lib
cmake_minimum_required(VERSION 3.18)
project(mean_shift_lib CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Setup CUDA
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(NOT DEFINED CMAKE_CUDA_STANDARD)
	set(CMAKE_CUDA_STANDARD 11)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

#set(CMAKE_CUDA_ARCHITECTURES 61)

# Library target
add_library(mean_shift SHARED
	ms_utils.h 
	helper_cuda.h 
	timer.hpp 
	include/mean_shift.h
	mean_shift.cpp
	include/cumean_shift.cuh
	cumean_shift_kernels.cuh
	cumean_shift_kernels.cu
	cumean_shift_kernels_experimental.cu
	cumean_shift.cu
	mean_shift_postprocessing.h
	mean_shift_postprocessing.cpp )
target_include_directories(mean_shift PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
#set_property( TARGET mean_shift PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mean_shift PROPERTIES PUBLIC_HEADER "include/cumean_shift.cuh;include/mean_shift.h")
target_compile_options(mean_shift PRIVATE $<$<CONFIG:Debug>:$<$<COMPILE_LANGUAGE:CUDA>:
	--generate-line-info
>>)

# Support installation of the library
include(GNUInstallDirs)
install( TARGETS mean_shift
         RUNTIME DESTINATION bin/mean-shift
         LIBRARY DESTINATION lib/mean-shift
         PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mean-shift )

