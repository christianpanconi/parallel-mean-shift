/**
 * Utilities
 */
#ifndef SRC_MEAN_SHIFT_MEAN_SHIFT_PY_MODULE_UTILS_H_
#define SRC_MEAN_SHIFT_MEAN_SHIFT_PY_MODULE_UTILS_H_

#include <Python.h>
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
/** These:
 * #define NO_IMPORT_ARRAY
 * #define PY_ARRAY_UNIQUE_SYMBOL meanshift_ARRAY_API
 *
 * seems not needed in header-only helpers because the
 * functions are supposed to be called in a proper initialized
 * numpy context.
 */
#include <numpy/arrayobject.h>

#include <memory>
#include <string>
#include <iostream>

static const size_t SIZE_SHORT = sizeof(short);
static const size_t SIZE_INT = sizeof(int);
static const size_t SIZE_LONG = sizeof(long);
static const size_t SIZE_USHORT = sizeof(unsigned short);
static const size_t SIZE_UINT = sizeof(unsigned int);
static const size_t SIZE_ULONG = sizeof(unsigned long);
static const size_t SIZE_FLOAT = sizeof(float);
static const size_t SIZE_DOUBLE = sizeof(double);
static const size_t SIZE_LONGDOUBLE = sizeof(long double);

template<typename... T>
static void throw_strexception( const std::string& fmt , T... args ){
	int size_s = std::snprintf( nullptr , 0 , fmt.c_str() , args ... ) + 1;
	if( size_s <= 0 ){ throw std::runtime_error("Error while formatting"); }
	auto size = static_cast<size_t>( size_s );
	std::unique_ptr<char[]> buf( new char[size] );
	std::snprintf( buf.get() , size , fmt.c_str() , args ... );
	std::string exstr = std::string( buf.get() , buf.get() + size );
	throw exstr;
}

static std::string point_to_str( float const* const p , unsigned int m ){
	std::string pstr = "[";
	for( int i=0 ; i<m ; i++ )
		pstr += std::to_string(p[i]) + (i==m-1?"]":" ");
	return pstr;
}

template<typename ND_T , typename C_T>
static C_T* __ndarray2D_to_carray( PyArrayObject* ndarray ){
	npy_intp *dims = PyArray_DIMS(ndarray);
	ND_T* nddata = static_cast<ND_T*>(PyArray_DATA(ndarray));
	C_T* carray = (C_T*) malloc(dims[0]*dims[1]*sizeof(C_T));

	for( int i=0 ; i<dims[0]*dims[1] ; i++ )
		carray[i] = static_cast<C_T>( nddata[i] );

	return carray;
}

/**
 * Returns a copy of the data member inside a 2-dimensional PyArray_Object,
 * as a C array of the type specified by the template parameter.
 * The data type of the PyArray_Object elements must be compatible with a primitive C data type.
 * Supported numpy data types:
 * 	numpy.short			| numpy.int16
 * 	numpy.ushort		| numpy.uint16
 * 	numpy.intc			| numpy.int32
 * 	numpy.uintc			| numpy.uint32
 * 	numpy.int_			| numpy.int64
 * 	numpy.uint			| numpy.uint64
 * 	numpy.single 		| numpy.float32
 * 	numpy.double 		| numpy.float64
 * 	numpy.longdouble 	| numpy.float128
 *
 * 	This method throws a string as exception if the passed ndarray is not 2-dimensional,
 * 	of if it does not contain elements of a supported data type.
 */
template<typename T>
static T* ndarray2D_to_carray( PyArrayObject* ndarray ){
	int ndim = PyArray_NDIM(ndarray);
	if( ndim != 2 )
		throw_strexception( "the ndarray must be 2-dimensional, but the passed ndarray has %d dimension(s)." ,
			ndim );

	PyArray_Descr *descr = PyArray_DESCR(ndarray);
//	int elsize = descr->elsize;
	const size_t elsize = descr->elsize;

	T* carray = nullptr;
	if( PyArray_ISNUMBER(ndarray) )
		if( PyArray_ISINTEGER(ndarray) ){
			if( PyArray_ISUNSIGNED(ndarray) ){
				if( elsize == SIZE_USHORT ) carray=__ndarray2D_to_carray<unsigned short , T>(ndarray);
				else if( elsize == SIZE_UINT ) carray=__ndarray2D_to_carray<unsigned int , T>(ndarray);
				else if( elsize == SIZE_ULONG ) carray=__ndarray2D_to_carray<unsigned long , T>(ndarray);
			}else{
				if( elsize == SIZE_SHORT ) carray=__ndarray2D_to_carray<short , T>(ndarray);
				else if( elsize == SIZE_INT ) carray=__ndarray2D_to_carray<int , T>(ndarray);
				else if( elsize == SIZE_LONG ) carray=__ndarray2D_to_carray<long , T>(ndarray);
			}
		}else if( PyArray_ISFLOAT(ndarray) ){
			if( elsize == SIZE_FLOAT ) carray=__ndarray2D_to_carray<float , T>(ndarray);
			else if( elsize == SIZE_DOUBLE ) carray=__ndarray2D_to_carray<double , T>(ndarray);
			else if( elsize == SIZE_LONGDOUBLE ) carray=__ndarray2D_to_carray<long double , T>(ndarray);
		}else
			throw_strexception( "invalid ndarray elements type, only integers and floats are supported.");
	else
		throw_strexception( "invalid ndarray elements type, not numeric." );

	if( carray == nullptr )
		throw_strexception( "unsupported data type, unsupported combination of kind='%c' and size=%d." ,
				descr->kind , elsize );

	return carray;
}

// Planar version
template<typename ND_T , typename C_T>
static C_T* __ndarray2D_to_carray_planar( PyArrayObject *ndarray ){
	npy_intp* dims = PyArray_DIMS(ndarray);
	unsigned int n=dims[0], m=dims[1];
	ND_T* nddata = static_cast<ND_T*>(PyArray_DATA(ndarray) );
	C_T* carray = new C_T[dims[0]*dims[1]];

	for( int i=0 ; i<n ; i++ ){
		for( int j=0 ; j<m ; j++ ){
			carray[j*n+i] = nddata[i*m+j];
		}
	}

	return carray;
}

template<typename T>
static T* ndarray2D_to_carray_planar( PyArrayObject *ndarray ){
	int ndim = PyArray_NDIM(ndarray);
	if( ndim != 2 )
		throw_strexception( "the ndarray must be 2-dimensional, but the passed ndarray has %d dimension(s)." ,
					ndim );

	PyArray_Descr* descr = PyArray_DESCR(ndarray);
	const size_t elsize = descr->elsize;
	T* carray = nullptr;

	if( PyArray_ISNUMBER(ndarray) ){
		if( PyArray_ISINTEGER(ndarray) ){
			if( PyArray_ISUNSIGNED(ndarray) ){
				if( elsize == SIZE_USHORT ){
					carray = __ndarray2D_to_carray_planar<unsigned short, T>(ndarray);
				}else if( elsize == SIZE_UINT ){
					carray = __ndarray2D_to_carray_planar<unsigned int, T>(ndarray);
				}else if( elsize == SIZE_ULONG){
					carray = __ndarray2D_to_carray_planar<unsigned long, T>(ndarray);
				}
			}else{
				if( elsize == SIZE_SHORT ){
					carray = __ndarray2D_to_carray_planar<short, T>(ndarray);
				}else if( elsize == SIZE_INT ){
					carray = __ndarray2D_to_carray_planar<int, T>(ndarray);
				}else if( elsize == SIZE_LONG ){
					carray = __ndarray2D_to_carray_planar<long, T>(ndarray);
				}
			}
		}else if( PyArray_ISFLOAT(ndarray) ){
			if( elsize == SIZE_FLOAT ){
				carray = __ndarray2D_to_carray_planar<float, T>(ndarray);
			}else if( elsize == SIZE_DOUBLE ){
				carray = __ndarray2D_to_carray_planar<double, T>(ndarray);
			}else if( elsize == SIZE_LONGDOUBLE ){
				carray = __ndarray2D_to_carray_planar<long double, T>(ndarray);
			}
		}
	}else
		throw_strexception( "invalid ndarray elements type, not numeric." );

	if( carray == nullptr )
		throw_strexception( "unsupported data type, unsupported combination of kind='%c' and size=%d." ,
				descr->kind , elsize );
	return carray;
}

#endif /* SRC_MEAN_SHIFT_MEAN_SHIFT_PY_MODULE_UTILS_H_ */
