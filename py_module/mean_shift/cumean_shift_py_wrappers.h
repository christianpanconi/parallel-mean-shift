#ifndef SRC_MEAN_SHIFT_CUMEAN_SHIFT_PY_WRAPPERS_H_
#define SRC_MEAN_SHIFT_CUMEAN_SHIFT_PY_WRAPPERS_H_

#include <Python.h>

PyObject* cu_ms_clustering(PyObject* self, PyObject* args);

PyObject* cu_ms_kernel_launch_info(PyObject* self, PyObject* args);

PyObject* get_CUDA_device_info( PyObject* self , PyObject* args );

#endif /* SRC_MEAN_SHIFT_CUMEAN_SHIFT_PY_WRAPPERS_H_ */
