#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "timer.hpp"
#include "helper_cuda.h"
#include "ms_utils.h"

#include "mean_shift_postprocessing.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "include/cumean_shift.cuh"
#include "cumean_shift_kernels.cuh"

namespace MeanShift{
namespace cuda{

namespace {
__host__ void launch_ms_kernel_simple(
		float* data , unsigned int n , unsigned int m ,
		float h , float tol , unsigned int max_iter ,
		float *conv_pts , unsigned int block_size ,
		cudaDeviceProp props , cudaStream_t stream ,
		cudaEvent_t ms_start , cudaEvent_t ms_end ){

	if( block_size == 0 )
		block_size = 64;
	dim3 blockSize( block_size );
	dim3 gridSize( (int)ceil(n/(float)blockSize.x) );
	std::size_t shmem_size = blockSize.x * m * sizeof( float );

	cudaEventRecord( ms_start , stream );
	_mean_shift_kernel<<< gridSize , blockSize , shmem_size , stream >>>
		( data , n , m , h , tol*tol , max_iter , conv_pts );
	cudaStreamSynchronize( stream );
	printLastCudaError( "_mean_shift_kernel_simple" );
	cudaEventRecord( ms_end , stream );
}

__host__ void launch_ms_kernel_tiled(
		float *data , unsigned int n , unsigned int m ,
		float h , float tol , unsigned int max_iter ,
		float *conv_pts , unsigned int block_size ,
		cudaDeviceProp props , cudaStream_t stream ,
		cudaEvent_t ms_start , cudaEvent_t ms_end ){

	if( block_size == 0 )
		block_size = 32;
	dim3 blockSize( block_size );
	dim3 gridSize( (int)ceil(n/(float)blockSize.x) );

	std::size_t shmem_size = 3 * m * blockSize.x * sizeof(float) + sizeof(int);

	cudaEventRecord( ms_start , stream );
	_mean_shift_kernel_tiled<<< gridSize , blockSize , shmem_size , stream >>>
		( data , n , m , h , tol , max_iter , conv_pts );
	cudaStreamSynchronize( stream );
	printLastCudaError( "_mean_shift_kernel_tiled" );
	cudaEventRecord( ms_end , stream );
}

__host__ void launch_ms_kernel_tiled_fixed(
		float *data , unsigned int n , unsigned int m ,
		float h , float tol , unsigned int max_iter ,
		float *conv_pts , unsigned int block_size ,
		cudaDeviceProp props , cudaStream_t stream ,
		cudaEvent_t ms_start , cudaEvent_t ms_end ){

	if( block_size == 0 )
		block_size = 32;
	dim3 blockSize( block_size );
	dim3 gridSize( (int)ceil(n/(float)blockSize.x) );

	std::size_t shmem_size = 2 * m * blockSize.x * sizeof(float) + sizeof(int);

	cudaEventRecord( ms_start , stream );
	_mean_shift_kernel_tiled_fixed<<< gridSize , blockSize , shmem_size , stream >>>
		( data , n , m , h , tol , max_iter , conv_pts );
	cudaStreamSynchronize( stream );
	printLastCudaError( "_mean_shift_kernel_tiled_fixed" );
	cudaEventRecord( ms_end , stream );
}

__host__ void launch_ms_kernel_exp(
		float * data, unsigned int n, unsigned int m,
		float h, float tol, unsigned int max_iter,
		float *conv_pts, unsigned int block_size,
		cudaDeviceProp props, cudaStream_t stream,
		cudaEvent_t ms_start, cudaEvent_t ms_end ){

	if( block_size == 0 )
		block_size = 32;
	dim3 blockSize( block_size );
	dim3 gridSize( (int)ceil(n/(float)blockSize.x) );

	// simple, double buffered
	std::size_t shmem_size = 2 * m * blockSize.x * sizeof(float) + sizeof(int);
	// simple packed access
//	std::size_t shmem_size = m * blockSize.x * sizeof(float);

	cudaEventRecord( ms_start , stream );
	_mean_shift_kernel_exp<<< gridSize , blockSize , shmem_size , stream >>>
		( data , n , m , h , tol , max_iter , conv_pts );
	cudaStreamSynchronize(stream);
	printLastCudaError( "_mean_shift_kernel_exp" );
	cudaEventRecord( ms_end , stream );
}

void (*kernel_launchers[])
		(float* , unsigned int , unsigned int ,
	     float , float , unsigned int ,
		 float* , unsigned int ,
		 cudaDeviceProp , cudaStream_t ,
		 cudaEvent_t , cudaEvent_t ) = {
	launch_ms_kernel_simple ,
	launch_ms_kernel_tiled ,
	launch_ms_kernel_tiled_fixed ,
	launch_ms_kernel_exp
};

}

__host__ float* cu_mean_shift(
		float* ms_time ,
		float* data , unsigned int n, unsigned int m,
		float h, float tol , unsigned int max_iter ,
		int kernel_type , unsigned int block_size ){

	cudaDeviceProp props;
	int devID = findCudaDevice( 0 , nullptr );
	checkCudaErrors( cudaGetDeviceProperties(&props , devID) );

	float *data_d, *conv_pts_d, *conv_pts;

	cudaStream_t stream1;
	cudaStreamCreate( &stream1 );

	cudaEvent_t ms_start, ms_end;
	cudaEventCreate( &ms_start );
	cudaEventCreate( &ms_end );

	cudaMalloc( &data_d , n*m*sizeof(float) );
	cudaMalloc( &conv_pts_d , n*m*sizeof(float) );
	conv_pts = new float[n*m];
	cudaMemcpyAsync( data_d , data , n*m*sizeof(float) , cudaMemcpyKind::cudaMemcpyHostToDevice , stream1 );

	(*kernel_launchers[kernel_type])(
		data_d , n , m , h , tol , max_iter ,
		conv_pts_d , block_size ,
		props , stream1 , ms_start , ms_end
	);

	cudaMemcpyAsync( conv_pts , conv_pts_d , n*m*sizeof(float) , cudaMemcpyKind::cudaMemcpyDeviceToHost , stream1 );
	cudaStreamSynchronize(stream1);

	cudaFree(data_d);
	cudaFree(conv_pts_d);

	if( ms_time != nullptr )
		cudaEventElapsedTime( ms_time , ms_start, ms_end );
	cudaEventDestroy(ms_start);
	cudaEventDestroy(ms_end);
	cudaStreamDestroy(stream1);

	return conv_pts;
}

__host__ std::vector<Cluster_t> cu_mean_shift_clustering(
		MSResult *result,
		float *data , unsigned int n , unsigned int m ,
		float h , float tol , unsigned int max_iter , float agg_th ,
		int kernel_type , unsigned int block_size ){

	cudaDeviceProp props;
	int devID = findCudaDevice( 0 , nullptr );
	checkCudaErrors( cudaGetDeviceProperties(&props, devID) );

	float *data_d, *conv_pts_d;
	float *conv_pts;

	cudaStream_t stream1;
	cudaStreamCreate( &stream1 );

	cudaEvent_t ms_start, ms_end;
	cudaEventCreate( &ms_start );
	cudaEventCreate( &ms_end );

	cudaMalloc( &data_d , n*m*sizeof(float) );
	cudaMalloc( &conv_pts_d , n*m*sizeof(float) );
	cudaMallocHost( &conv_pts , n*m*sizeof(float) );
	cudaMemcpyAsync( data_d , data , n*m*sizeof(float) , cudaMemcpyKind::cudaMemcpyHostToDevice , stream1 );

	// Kernel launch
	(*kernel_launchers[kernel_type])(
		data_d , n , m , h , tol , max_iter ,
		conv_pts_d , block_size ,
		props , stream1 , ms_start , ms_end );

	cudaMemcpyAsync( conv_pts , conv_pts_d , n*m*sizeof(float) , cudaMemcpyKind::cudaMemcpyDeviceToHost , stream1 );
	cudaStreamSynchronize( stream1 );

	cudaFree( data_d );
	cudaFree( conv_pts_d );

	c8::Timer timer;
	std::vector<Cluster_t> clusters;

	// Form clusters
	timer.start();
	clusters = _form_clusters_planar(conv_pts, n, m, agg_th);
	timer.stop();

	cudaFreeHost(conv_pts);
	if( result != nullptr ){
		cudaEventElapsedTime( &(result->mean_shift_time) , ms_start , ms_end );
		result->clusters_formation_time = timer.elapsed<std::chrono::nanoseconds>()/1000000.0;
	}
	cudaEventDestroy( ms_start ); cudaEventDestroy( ms_end );
	cudaStreamDestroy( stream1 );

	return clusters;
}

void get_kernel_launch_info(
		MeanShift::cuda::KernelLaunchInfo* info,
		cudaDeviceProp props , int kernel_type ,
		dim3 blockSize , unsigned int shmem_size ){

	// CUDA 12 / CC 8.9
	int REGS_PER_THREAD[4] = {
		40 , // simple
		56 , // tiled
		48 , // tiled_fixed
		40   // experimental
	};

	// CUDA 11.7 / CC 6.1
//	int REGS_PER_THREAD[4] = {
//		32 , // simple
//		54 , // tiled
//		48 , // tiled_fixed
//		50   // experimental
//	};

	info->maxWarpsPerSM = (unsigned int)(props.maxThreadsPerMultiProcessor/(float)props.warpSize);
	info->warpsPerBlock = (unsigned int)ceil(blockSize.x/(float)props.warpSize);
	info->regsPerThread = REGS_PER_THREAD[kernel_type];
	info->regsPerBlock = info->regsPerThread * blockSize.x;
	info->regsBlocksPerSM = (unsigned int)floor(props.regsPerMultiprocessor / (float)info->regsPerBlock);
//	if( info.regsConcurrentBlocksPerSM < props.maxBlocksPerMultiProcessor )
		// # of blocks limited to info.regsConcurrentBlocksPerSM by register limits
	info->sharedMemPerBlock = shmem_size;
	info->sharedMemBlocksPerSM = (unsigned int)floor(props.sharedMemPerMultiprocessor/(float)shmem_size );
//	if( info.sharedMemConcurrentBlocksPerSM < props.maxBlocksPerMultiProcessor )
		// # of blocks limited to info.sharedMemConcurrentBlocksPerSM by shared memory limits
	if( info->regsBlocksPerSM < info->sharedMemBlocksPerSM )
		info->blocksPerSM = info->regsBlocksPerSM;
	else
		info->blocksPerSM = info->sharedMemBlocksPerSM;

	if( info->blocksPerSM > props.maxBlocksPerMultiProcessor ) // check against HW limits
		info->blocksPerSM = props.maxBlocksPerMultiProcessor;

	info->warpsPerSM = (unsigned int) floor( (info->blocksPerSM*blockSize.x)/(float)props.warpSize );
	info->occupancy = info->warpsPerSM / (float) info->maxWarpsPerSM;
}

} // cuda namespace
} // MeanShift namespace

