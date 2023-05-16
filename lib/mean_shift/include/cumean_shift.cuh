/**
 * CUMeanShift custom lib
 */
#ifndef LIB_MEAN_SHIFT_INCLUDE_CUMEAN_SHIFT_CUH_
#define LIB_MEAN_SHIFT_INCLUDE_CUMEAN_SHIFT_CUH_

#include <cuda_runtime.h>
#include <vector>

#include "../mean_shift_postprocessing.h"

namespace MeanShift{
namespace cuda {

const int KERNEL_TYPE_SIMPLE = 0;
const int KERNEL_TYPE_TILED = 1;
const int KERNEL_TYPE_TILED_FIXED = 2;
const int KERNEL_TYPE_EXPERIMENTAL = 3;

typedef struct KernelLaunchInfo {
	unsigned int maxWarpsPerSM;
	unsigned int warpsPerBlock;
	unsigned int regsPerThread;
	unsigned int regsPerBlock;
	unsigned int sharedMemPerBlock;

	unsigned int regsBlocksPerSM;
	unsigned int sharedMemBlocksPerSM;
	unsigned int blocksPerSM;
	unsigned int warpsPerSM;

	float occupancy;
} KernelLaunchInfo;

void get_kernel_launch_info( KernelLaunchInfo* info,
		cudaDeviceProp props , int kernel_type ,
		dim3 blockSize , unsigned int shmem_size );

/**
 * ================================
 * MEAN SHIFT FUNCTION
 * ================================
 */
__host__ float* cu_mean_shift(
		float* ms_time ,
		float* data , unsigned int n, unsigned int m,
		float h, float tol , unsigned int max_iter ,
		int kernel_type , unsigned int block_size=0 );

/**
 * ================================
 * CLUSTERING FUNCTION
 * ================================
 */
__host__ std::vector<Cluster_t> cu_mean_shift_clustering(
		MSResult *result ,
		float *data , unsigned int n , unsigned int m ,
		float h , float tol , unsigned int max_iter , float agg_th ,
		int kernel_type , unsigned int block_size=0 );

} // end cuda namespace
} // end MeanShift namespace
#endif /* LIB_MEAN_SHIFT_INCLUDE_CUMEAN_SHIFT_CUH_ */
