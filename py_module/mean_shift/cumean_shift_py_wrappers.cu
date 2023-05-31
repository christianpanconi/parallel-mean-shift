#include <Python.h>
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif /*NPY_NO_DEPRECATED_API*/
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL meanshift_ARRAY_API
#include <numpy/arrayobject.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cumean_shift_py_wrappers.h"
#include "mean_shift_py_module_utils.h"
#include "../../lib/mean_shift/include/cumean_shift.cuh"
#include "../../lib/mean_shift/helper_cuda.h"

#include <cmath>
#include <chrono>
#include "timer.hpp"


PyObject* cu_mean_shift( PyObject* self , PyObject* args ){
	c8::Timer timer;

	PyArrayObject* data_ndarray;
	float h, tol;
	unsigned int max_iter;
	int kernel_type;
	unsigned int block_size;

	// Parse py args
	if( !PyArg_ParseTuple( args , "O!ffIiI" ,
			&PyArray_Type , &data_ndarray , &h , &tol , &max_iter ,
			&kernel_type , &block_size ) )
		return nullptr;

	// Allocate data in planar format
	float *fdata;
	try{
		fdata = ndarray2D_to_carray_planar<float>(data_ndarray);
	}catch( std::string& exstr ){
		PyErr_SetString( PyExc_ValueError , exstr.c_str() );
		return nullptr;
	}

	npy_intp *dims = PyArray_DIMS(data_ndarray);
	unsigned int n = dims[0] , m = dims[1];

	float* conv_pts;
	float ms_time;
	try{
		timer.start();
		// Call Mean Shift function
		conv_pts = MeanShift::cuda::cu_mean_shift( &ms_time ,
			fdata, n, m, h, tol, max_iter,
			kernel_type, block_size );
		timer.stop();
	}catch( std::string& exstr ){
		PyErr_SetString( PyExc_ValueError , exstr.c_str() );
		delete fdata;
		return Py_None;
	}

	//Build result
	float *conv_pts_packed = new float[n*m];
	for( int j=0 ; j<m ; j++ ){
		for( int i=0 ; i<n ; i++ ){
			conv_pts_packed[i*m+j] = conv_pts[j*n+i];
		}
	}
	delete conv_pts;

	PyObject* conv_pts_ndarray = PyArray_SimpleNewFromData(
		2 , dims , NPY_TYPES::NPY_FLOAT , conv_pts_packed );

	PyObject* result = PyDict_New();
	PyDict_SetItemString( result , "conv_pts" , conv_pts_ndarray );
	PyDict_SetItemString( result , "mean_shift_time", PyFloat_FromDouble(ms_time) );
	return result;
}

/**
 * Wrapper for the CUDA Mean Shift clustering implementation.
 */
PyObject* cu_ms_clustering(PyObject* self, PyObject* args){
	c8::Timer timer;

	PyArrayObject* data_ndarray;
	float h, tol, agg_th;
	unsigned int max_iter;
	int kernel_type;
	unsigned int block_size;

	// Parse python args
	if( !PyArg_ParseTuple( args , "O!fffIiI" ,
			&PyArray_Type , &data_ndarray ,
			&h, &tol, &agg_th, &max_iter ,
			&kernel_type , &block_size ) )
		return nullptr;

	// Allocate data in planar format
	float *fdata;
	try{
		fdata = ndarray2D_to_carray_planar<float>(data_ndarray);
	}catch( std::string& exstr ){
		PyErr_SetString( PyExc_ValueError , exstr.c_str() );
		return nullptr;
	}

	npy_intp *dims = PyArray_DIMS(data_ndarray);
	unsigned int n=dims[0], m=dims[1];

	// Call clustering function
	std::vector<Cluster_t> clusters;
	MSResult ms_result;
	try{
		timer.start();
		clusters = MeanShift::cuda::cu_mean_shift_clustering(
			&ms_result , fdata , n , m ,
			h , tol , max_iter , agg_th ,
			kernel_type , block_size );
		timer.stop();
	}catch( std::string& exstr ){
		PyErr_SetString( PyExc_RuntimeError , exstr.c_str() );
		delete fdata;
		return Py_None;
	}
	delete fdata;

	// Build result
	PyObject *clusters_list = PyList_New(clusters.size());
	for( int c=0 ; c<clusters.size() ; c++ ){
		PyObject *cluster_list = PyList_New(clusters[c].pts_indices.size());
			for( int ci=0 ; ci < clusters[c].pts_indices.size() ; ci++ )
				PyList_SET_ITEM( cluster_list , ci , PyLong_FromLong(clusters[c].pts_indices[ci]) );
		PyList_SET_ITEM( clusters_list , c , cluster_list );
	}

	float *centroids = new float[clusters.size()*m];
	for( int i=0 ; i<clusters.size() ; i++ ){
		for( int j=0 ; j<m ; j++ ){
			centroids[i*m+j] = clusters[i].centroid[j];
		}
	}
	npy_intp *centroids_dims = new npy_intp[2];
	centroids_dims[0] = clusters.size();
	centroids_dims[1] = m;
	PyObject* centroids_ndarray = PyArray_SimpleNewFromData(
		2 , centroids_dims , NPY_TYPES::NPY_FLOAT , centroids );

	PyObject *result = PyDict_New();
	PyDict_SetItemString( result , "clusters" , clusters_list );
	PyDict_SetItemString( result , "centroids" , centroids_ndarray );
	PyDict_SetItemString( result , "mean_shift_time" ,
		PyFloat_FromDouble(ms_result.mean_shift_time ) );
	PyDict_SetItemString( result , "clusters_formation_time" ,
		PyFloat_FromDouble(ms_result.clusters_formation_time) );
	PyDict_SetItemString( result , "elapsed_ms" ,
		PyFloat_FromDouble( timer.elapsed<std::chrono::nanoseconds>()/1000000.0 ) );
//		PyLong_FromUnsignedLongLong(timer.elapsed<std::chrono::milliseconds>()) );

	for( int i=0 ; i<clusters.size() ; i++ )
		delete clusters[i].centroid;
	return result;
}

/**
 * Provides informations about CUDA Mean Shift kernel launch.
 */
PyObject* cu_ms_kernel_launch_info( PyObject* self , PyObject* args ){
	int kernel_type;
	unsigned int m;
	unsigned int block_size;
	unsigned int tile_size;

	if( !PyArg_ParseTuple( args , "iII" ,
			&kernel_type , &m ,
			&block_size ) )
		return Py_None;

	cudaDeviceProp props;
	int devID = findCudaDevice( 0 , nullptr ); // Finds best CUDA capable device
	checkCudaErrors( cudaGetDeviceProperties(&props, devID) );

	dim3 blockSize( block_size );
	unsigned int shmem_size;

	if( kernel_type == MeanShift::cuda::KERNEL_TYPE_SIMPLE ){
		tile_size = 0;
		shmem_size = blockSize.x * m * sizeof(float);
	}

	if( kernel_type == MeanShift::cuda::KERNEL_TYPE_TILED ){
		if( tile_size == 0 )
			tile_size = floorf(((props.sharedMemPerMultiprocessor/(float)props.maxBlocksPerMultiProcessor)
						- (m*blockSize.x*sizeof(float)+sizeof(int)))/(m*sizeof(float)));
		tile_size = block_size;
		shmem_size = 3*blockSize.x*m*sizeof(float) + sizeof(int);
	}

	MeanShift::cuda::KernelLaunchInfo info;
	MeanShift::cuda::get_kernel_launch_info( &info , props, kernel_type, blockSize, shmem_size );

	PyObject* pyKernelInfo = PyDict_New();
	PyDict_SetItemString( pyKernelInfo , "block_size" , PyLong_FromUnsignedLong( blockSize.x ) );
	PyDict_SetItemString( pyKernelInfo , "warps_per_block" , PyLong_FromUnsignedLong(info.warpsPerBlock) );
	PyDict_SetItemString( pyKernelInfo , "tile_size" , PyLong_FromUnsignedLong(tile_size) );
	PyDict_SetItemString( pyKernelInfo , "max_warps_per_SM" , PyLong_FromUnsignedLong(info.maxWarpsPerSM) );
	PyDict_SetItemString( pyKernelInfo , "regs_per_thread" , PyLong_FromUnsignedLong(info.regsPerThread) );
	PyDict_SetItemString( pyKernelInfo , "regs_per_block" , PyLong_FromUnsignedLong(info.regsPerBlock) );
	PyDict_SetItemString( pyKernelInfo , "shmem_per_block" , PyLong_FromUnsignedLong(info.sharedMemPerBlock) );
	PyDict_SetItemString( pyKernelInfo , "regs_blocks_per_SM" , PyLong_FromUnsignedLong(info.regsBlocksPerSM) );
	PyDict_SetItemString( pyKernelInfo , "shmem_blocks_per_SM" , PyLong_FromUnsignedLong(info.sharedMemBlocksPerSM) );
	PyDict_SetItemString( pyKernelInfo , "blocks_per_SM" , PyLong_FromUnsignedLong(info.blocksPerSM) );
	PyDict_SetItemString( pyKernelInfo , "warps_per_SM" , PyLong_FromUnsignedLong(info.warpsPerSM) );
	PyDict_SetItemString( pyKernelInfo , "occupancy" , PyFloat_FromDouble(info.occupancy) );

	return pyKernelInfo;
}

/**
 * Provides informations about a CUDA capable device.
 */
PyObject* get_CUDA_device_info( PyObject* self , PyObject* args ){

	int deviceID;
	if( !PyArg_ParseTuple( args , "i" , &deviceID ) )
		return Py_None;

	if(deviceID < 0)
		deviceID = findCudaDevice( 0 , nullptr );
	else{
		deviceID = gpuDeviceInit(deviceID);
		if( deviceID < 0 ){
			return Py_None;
		}
	}

	cudaDeviceProp props;
	checkCudaErrors( cudaGetDeviceProperties(&props , deviceID) );

	PyObject* pyGPUInfo = PyDict_New();
	PyDict_SetItemString( pyGPUInfo , "name" , Py_BuildValue( "s" , props.name ) );
	PyDict_SetItemString( pyGPUInfo , "device_id" , PyLong_FromLong( deviceID ) );
	PyDict_SetItemString( pyGPUInfo , "major" , PyLong_FromLong( props.major ) );
	PyDict_SetItemString( pyGPUInfo , "minor" , PyLong_FromLong( props.minor ) );
	PyDict_SetItemString( pyGPUInfo , "arch_name" , Py_BuildValue( "s" , _ConvertSMVer2ArchName(props.major , props.minor) ) );
	PyDict_SetItemString( pyGPUInfo , "cuda_cores_per_SM" , PyLong_FromLong( _ConvertSMVer2Cores(props.major, props.minor) ) );
	PyDict_SetItemString( pyGPUInfo , "pci_bus_id" , PyLong_FromLong( props.pciBusID ) );
	PyDict_SetItemString( pyGPUInfo , "pci_device_id" , PyLong_FromLong( props.pciDeviceID ) );
	PyDict_SetItemString( pyGPUInfo , "pci_domain_id" , PyLong_FromLong( props.pciDomainID ) );
	PyDict_SetItemString( pyGPUInfo , "max_blocks_per_SM" , PyLong_FromLong(props.maxBlocksPerMultiProcessor) );
	PyDict_SetItemString( pyGPUInfo , "max_threads_per_block" , PyLong_FromLong(props.maxThreadsPerBlock) );
	PyDict_SetItemString( pyGPUInfo , "max_threads_per_SM" , PyLong_FromLong(props.maxThreadsPerMultiProcessor) );
	PyDict_SetItemString( pyGPUInfo , "max_grid_size" , Py_BuildValue("(i,i,i)" ,
		props.maxGridSize[0] , props.maxGridSize[1] , props.maxGridSize[2] ) );
	PyDict_SetItemString( pyGPUInfo , "max_threads_dim" , Py_BuildValue("(i,i,i)" ,
		props.maxThreadsDim[0] , props.maxThreadsDim[1] , props.maxThreadsDim[2] ) );
	PyDict_SetItemString( pyGPUInfo , "SM_count" , PyLong_FromLong(props.multiProcessorCount) );
	PyDict_SetItemString( pyGPUInfo , "regs_per_block" , PyLong_FromLong(props.regsPerBlock) );
	PyDict_SetItemString( pyGPUInfo , "regs_per_SM" , PyLong_FromLong(props.regsPerMultiprocessor) );
	PyDict_SetItemString( pyGPUInfo , "shmem_per_block" , PyLong_FromSize_t(props.sharedMemPerBlock) );
	PyDict_SetItemString( pyGPUInfo , "shmem_per_SM" , PyLong_FromSize_t(props.sharedMemPerMultiprocessor) );
	PyDict_SetItemString( pyGPUInfo , "total_global_mem" , PyLong_FromSize_t(props.totalGlobalMem) );
	PyDict_SetItemString( pyGPUInfo , "total_const_mem" , PyLong_FromSize_t(props.totalConstMem) );
	PyDict_SetItemString( pyGPUInfo , "warp_size" , PyLong_FromLong(props.warpSize) );

	return pyGPUInfo;
}


