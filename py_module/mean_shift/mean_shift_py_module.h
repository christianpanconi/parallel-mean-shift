/**
 * Mean shift module definition
 */
#ifndef SRC_MEAN_SHIFT_MEAN_SHIFT_PY_MODULE_H_
#define SRC_MEAN_SHIFT_MEAN_SHIFT_PY_MODULE_H_

#include <Python.h>

#include <stdio.h>
#include "cumean_shift_py_wrappers.h"
#include "mean_shift_py_wrappers.h"

// --- Module def
static PyMethodDef methods[] = {
	{"ms_clustering", ms_clustering, METH_VARARGS , "Sequential mean shift clustering."},
	{"cu_mean_shift" , cu_mean_shift , METH_VARARGS , "CUDA mean shift procedure."},
	{"cu_ms_clustering" , cu_ms_clustering , METH_VARARGS , "CUDA mean shift clustering."},
	{"cu_ms_kernel_launch_info" , cu_ms_kernel_launch_info , METH_VARARGS , "Provides info om MS kernel launch."},
	{"get_CUDA_device_info" , get_CUDA_device_info , METH_VARARGS , "" } ,
	{NULL, NULL, 0, NULL}
};

static PyModuleDef module = {
	PyModuleDef_HEAD_INIT,
	"_mean_shift",
	"_mean_shift",
	-1,
	methods
};

PyMODINIT_FUNC PyInit__mean_shift(void);

#endif /* SRC_MEAN_SHIFT_MEAN_SHIFT_PY_MODULE_H_ */
