/**
 * Python wrappers definitions for the C++ functions
 */
#ifndef SRC_MEAN_SHIFT_MEAN_SHIFT_PY_WRAPPERS_H_
#define SRC_MEAN_SHIFT_MEAN_SHIFT_PY_WRAPPERS_H_

#include <Python.h>

PyObject* cu_mean_shift( PyObject* self , PyObject* args );

PyObject* ms_clustering(PyObject* self, PyObject* args);

#endif /* SRC_MEAN_SHIFT_MEAN_SHIFT_PY_WRAPPERS_H_ */
