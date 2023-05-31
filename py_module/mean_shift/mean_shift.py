import mean_shift._mean_shift as ms
import numpy as np

MSKernelType = {
    "simple": 0,
    "tiled": 1
}
def __get_kernel_type_id( kernel_type: str ):
    if kernel_type not in MSKernelType:
        raise ValueError( "Unknown kernel type '{0}'. Allowed kernel types: {1}"
                         .format( kernel_type, list(MSKernelType.keys()) ))
    return MSKernelType[kernel_type]

# SEQUENTIAL MEAN SHIFT
def ms_clustering( data: np.ndarray , starts: np.ndarray=None , h: float=0.005 , tol: float=0.001 , agg_th: float=0.02 , max_iter: int=100 ):
    if starts is None:
        starts = data
    return ms.ms_clustering(starts , data , h , tol , agg_th , max_iter )

# CUDA MEAN SHIFT
def cu_mean_shift( data: np.ndarray , h: float=0.005 , tol: float=0.001 , max_iter: int=100 , kernel_type: str="simple" , block_size: int=0 ):
    if kernel_type not in MSKernelType:
        raise ValueError( "Unknown kernel type '{0}'. Allowed kernel types: {1}"
                          .format( kernel_type , list(MSKernelType.keys()) ) )
    kernel_type_id = MSKernelType[kernel_type]
    return ms.cu_mean_shift( data , h , tol , max_iter , kernel_type_id , block_size )

def cu_ms_clustering( data: np.ndarray , h: float=0.005 , tol: float=0.001 , agg_th: float=0.02 , max_iter: int=100 , kernel_type: str="simple" , block_size: int=0 ):
    if kernel_type not in MSKernelType:
        raise ValueError( "Unknown kernel type '{0}'. Allowed kernel types: {1}"
                          .format( kernel_type , list(MSKernelType.keys()) ) )
    kernel_type_id = MSKernelType[kernel_type]
    return ms.cu_ms_clustering( data , h , tol , agg_th , max_iter , kernel_type_id , block_size )

def cu_ms_kernel_info( kernel_type: str , n_dims: int , block_size: int ):
    assert n_dims >= 1 , "n_dims must be >= 1, but n_dims={0}".format(n_dims)
    assert block_size >= 1 , "block_size must be >= 1, but block_size={0}".format(block_size)
    kernel_type_id = __get_kernel_type_id(kernel_type)
    return ms.cu_ms_kernel_launch_info( kernel_type_id , n_dims , block_size )

def get_CUDA_device_info( device_id: int=-1 ):
    device_info = ms.get_CUDA_device_info(device_id)
    if device_info is None:
        raise RuntimeError( "Failed to init device with id ".format(device_id) )
    return device_info