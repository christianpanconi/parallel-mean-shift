/**
 * Mean shift module implementation
 */
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL meanshift_ARRAY_API
#include <numpy/arrayobject.h>

#include "mean_shift_py_module.h"

PyMODINIT_FUNC PyInit__mean_shift(void){
	import_array();
	return PyModule_Create(&module);
}
