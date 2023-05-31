/**
 * CUMeanShift kernels implementations
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cumean_shift_kernels.cuh"

/**
 * SIMPLE mean shift kernel
 *
 * CC 6.1: 32 regs/thread
 * CC 8.9: 40 regs/thread
 *
 * Regs/thread: 32
 * Shared mem required: blockDim.x*m*sizeof(float) bytes
 */
__global__ void _mean_shift_kernel(
		float *data , unsigned int n , unsigned int m ,
		float h , float sq_tol , unsigned int max_iter ,
		float *out ){

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	extern __shared__ float shbuf[];

	float g_val , g_acc, msv_sqnorm=0;
	float *in = data;

	bool thread_converged=false;

	if( tid < n ){
		for( int s=0 ; s<max_iter ; s++ ){
			for( int j=0 ; j<m ; j++ )
				shbuf[j*blockDim.x+threadIdx.x] = 0;

			// compute g function accumulation
			g_acc = 0;
			for( int i=0 ; i<n ; i++ ){
				g_val = 0;
				for( int j=0 ; j<m ; j++ )
//					g_val += powf( in[j*n+tid]-data[j*n+i] , 2.0 );
					g_val += (in[j*n+tid]-data[j*n+i])*(in[j*n+tid]-data[j*n+i]);
				g_val = expf( -0.5 * g_val/h );
				if( tid == i ) g_val = 0;
				g_acc += g_val;
				for( int j=0 ; j<m ; j++ )
					shbuf[j*blockDim.x+threadIdx.x] += g_val * data[j*n+i];
			}

			// compute shifted points and mean shift vector norm
			msv_sqnorm = 0;
			for( int j=0 ; j<m ; j++ ){
				shbuf[j*blockDim.x+threadIdx.x] /= g_acc;
				msv_sqnorm += powf( shbuf[j*blockDim.x+threadIdx.x]-in[j*n+tid] , 2.0 );
			}

			if( msv_sqnorm > sq_tol ){
				for( int j=0 ; j<m ; j++ ) // copy shared buffer to output if not converged
					out[j*n+tid] = shbuf[j*blockDim.x+threadIdx.x];
			}else{
				thread_converged = true;
				for( int j=0 ; j<m ; j++ ) // copy input to output if converged
					out[j*n+tid] = in[j*n+tid];
			}

			in = out; // After 1st iteration the input is always the points shifted by the previous iteration
			if( thread_converged )
				break;
		}
	}

}

/**
 * Tiled, tile_size:	blockDim.x
 * shared buffers: 		tile, in, out.
 *
 * CC 6.1: 48 regs/thread
 * CC 8.9: 48 regs/thread
 */
__global__ void _mean_shift_kernel_tiled(
	float* data, unsigned int n, unsigned int m,
	float h, float tol, unsigned int max_iter ,
	float *out){

	extern __shared__ float shbuf[];
	float *inbuf = shbuf + blockDim.x*m;
	float *outbuf = inbuf + blockDim.x*m;
	int *convergence_counter = reinterpret_cast<int*>(outbuf+blockDim.x*m);

	unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float g_val, g_acc, msv_sqnorm;

	bool thread_converged = false;
	if( threadIdx.x == 0)
		*convergence_counter = 0;
	__syncthreads();

	if( tid >= n ){
		thread_converged = true;
		atomicAdd( convergence_counter , 1 );
	}
	__syncthreads();

	if( tid < n )
		for( int j=0 ; j<m ; j++ ) // init inbuf
			inbuf[j*blockDim.x+threadIdx.x] = data[j*n+tid];
	__syncthreads();

	for( int s=0 ; s<max_iter && (*convergence_counter)<blockDim.x ; s++ ){
		for( int j=0 ; j<m ; j++ ) // reset outbuf
			outbuf[j*blockDim.x + threadIdx.x] = 0;
		g_acc = 0;

		for( int t=0 ; t<ceilf(n/(float)blockDim.x) ; t++ ){ //tiling loop
			if( t*blockDim.x + threadIdx.x < n ){
				for( int j=0 ; j<m ; j++ )
					shbuf[j*blockDim.x+threadIdx.x] = data[j*n + t*blockDim.x + threadIdx.x];
			}
			__syncthreads();

			if( !thread_converged ){
				for( int i=0 ; i<blockDim.x && t*blockDim.x+i < n ; i++ ){
					g_val=0;
					for( int j=0 ; j<m ; j++ ){
						g_val += (inbuf[j*blockDim.x+threadIdx.x]-shbuf[j*blockDim.x+i])*
								 (inbuf[j*blockDim.x+threadIdx.x]-shbuf[j*blockDim.x+i]);
					}
					g_val = expf( -0.5*g_val/h );
					if( tid == t*blockDim.x+i ) g_val = 0;
					g_acc += g_val;
					for( int j=0 ; j<m ; j++ )
						outbuf[j*blockDim.x+threadIdx.x] += g_val * shbuf[j*blockDim.x+i];
				}
			}
			__syncthreads();
		}

		if( !thread_converged ){
			msv_sqnorm = 0;
			for( int j=0 ; j<m ; j++ ){
				outbuf[j*blockDim.x+threadIdx.x] /= g_acc;
				msv_sqnorm += powf( outbuf[j*blockDim.x+threadIdx.x] - inbuf[j*blockDim.x+threadIdx.x] , 2.0 );
			}

			if( msv_sqnorm > tol*tol ){
				for( int j=0 ; j<m ; j++ )
					inbuf[j*blockDim.x+threadIdx.x] = outbuf[j*blockDim.x+threadIdx.x];
			}else{
				thread_converged = true;
				atomicAdd(convergence_counter , 1);
			}
		}
		__syncthreads();
	}

	for( int j=0 ; j<m ; j++ ) // copy from inbuf to output
		out[j*n+tid] = inbuf[j*blockDim.x+threadIdx.x];

}
