/**
 * (Static) Utilities
 */
#ifndef LIB_MEAN_SHIFT_MS_UTILS_H_
#define LIB_MEAN_SHIFT_MS_UTILS_H_

#include <string>
#include <memory>
#include <iostream>

#include "mean_shift_postprocessing.h"

static std::string p2str( float const * const p , const unsigned int m ){
	std::string pstr = "[";
	for( int i=0 ; i<m ; i++ )
		pstr += std::to_string(p[i]) + (i==m-1?"]":" ");
	return pstr;
}

static std::string p2str_planar( float const* const pts, unsigned int n, unsigned int m, unsigned int i){
	std::string pstr = "[";
	for( int j=0 ; j<m ; j++ )
		pstr += std::to_string(pts[j*m+i]) + (j==m-1?"]":", ");
	return pstr;
}

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

#endif /* LIB_MEAN_SHIFT_MS_UTILS_H_ */
