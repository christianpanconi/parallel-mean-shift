#ifndef LIB_MEAN_SHIFT_CUMEAN_SHIFT_KERNELS_CUH_
#define LIB_MEAN_SHIFT_CUMEAN_SHIFT_KERNELS_CUH_

#include <cuda_runtime.h>

/**
 * ================================
 * CUDA KERNELS
 * ================================
 */

// Simple
__global__ void _mean_shift_kernel(
		float *data , unsigned int n , unsigned int m ,
		float h , float sq_tol , unsigned int max_iter ,
		float *out );

// Tiled
__global__ void _mean_shift_kernel_tiled(
		float *data , unsigned int n, unsigned int m ,
		float h , float tol , unsigned int max_iter ,
		float *out );


#endif /* LIB_MEAN_SHIFT_CUMEAN_SHIFT_KERNELS_CUH_ */
