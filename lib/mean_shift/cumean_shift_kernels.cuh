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
//		const float* __restrict__ data,
		float *data , unsigned int n , unsigned int m ,
		float h , float sq_tol , unsigned int max_iter ,
		float *out );

__global__ void _mean_shift_kernel_tiled(
//		const float* __restrict__ data,
		float *data , unsigned int n, unsigned int m ,
		float h , float tol , unsigned int max_iter ,
		float *out );

__global__ void _mean_shift_kernel_tiled_fixed(
		float *data , unsigned int n, unsigned int m ,
		float h , float tol , unsigned int max_iter ,
		float *out );

// Experimental kernels
__global__ void _mean_shift_kernel_exp(
	const float* __restrict__ data,
//	float* data,
	unsigned int n, unsigned int m,
	float h, float tol, unsigned int max_iter ,
	float *out);


#endif /* LIB_MEAN_SHIFT_CUMEAN_SHIFT_KERNELS_CUH_ */
