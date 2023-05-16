/**
 * Python wrappers impl. for the C++ functions.
 */
#include <cstdio>
#include <iostream>

#include <Python.h>
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif /*NPY_NO_DEPRECATED_API*/
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL meanshift_ARRAY_API
#include <numpy/arrayobject.h>

#include "mean_shift_py_wrappers.h"
#include "mean_shift_py_module_utils.h"
#include "../../lib/mean_shift/include/mean_shift.h"
//#include "../../lib/mean_shift/mean_shift_postprocessing.h"
#include "timer.hpp"

#include <chrono>

/**
 * Wrapper for the SEQUENTIAL Mean Shift clustering implementation.
 */
PyObject* ms_clustering(PyObject* self, PyObject* args){
	PyArrayObject* data_ndarray;
	PyArrayObject* starts_ndarray;
	float h , tol, agg_th;
	unsigned int max_iter;

	if( !PyArg_ParseTuple(args, "O!O!fffI" ,
			&PyArray_Type , &starts_ndarray ,
			&PyArray_Type , &data_ndarray ,
			&h , &tol , &agg_th , &max_iter ) )
		return nullptr;

	float *fdata , *fstarts;
	try{
		fdata = ndarray2D_to_carray<float>(data_ndarray);
		if( starts_ndarray == data_ndarray )
			fstarts = fdata;
		else
			fstarts = ndarray2D_to_carray<float>(starts_ndarray);
	}catch( std::string& exstr ){
		PyErr_SetString( PyExc_ValueError , exstr.c_str() );
		return nullptr;
	}

	npy_intp* data_dims = PyArray_DIMS(data_ndarray);
	unsigned int n=data_dims[0], m=data_dims[1];
	npy_intp* starts_dims = PyArray_DIMS(starts_ndarray);
	unsigned int n_start = starts_dims[0];

	c8::Timer timer;
	MSResult ms_result;
	timer.start();
	std::vector<Cluster_t> clusters = MeanShift::seq::ms_clustering(
		&ms_result ,
		fstarts , fdata , n_start , n , m ,
		h , tol , agg_th , max_iter
	);
	timer.stop();

	if( fstarts != fdata )
		delete fstarts;
	delete fdata;

	// Build result
	PyObject *clusters_list = PyList_New(clusters.size());
	for( int c=0 ; c < clusters.size() ; c ++ ){
		PyObject *cluster_list = PyList_New(clusters[c].pts_indices.size());
		for( int ci=0 ; ci < clusters[c].pts_indices.size() ; ci++ ){
			PyList_SET_ITEM(cluster_list, ci , PyLong_FromUnsignedLong(clusters[c].pts_indices[ci]) );
		}
		PyList_SET_ITEM(clusters_list , c , cluster_list );
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
	PyDict_SetItemString(result, "clusters", clusters_list );
	PyDict_SetItemString(result, "centroids" , centroids_ndarray );
	PyDict_SetItemString(result , "mean_shift_time" ,
		PyFloat_FromDouble( ms_result.mean_shift_time ) );
	PyDict_SetItemString(result , "clusters_formation_time" ,
			PyFloat_FromDouble( ms_result.clusters_formation_time ) );
	PyDict_SetItemString(result, "elapsed_ms" ,
		PyLong_FromUnsignedLongLong(timer.elapsed<std::chrono::milliseconds>()) );

	return result;
}




