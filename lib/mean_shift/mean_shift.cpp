/**
 * Mean shift custom lib implementation
 */

#include "include/mean_shift.h"

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <string.h>
#include <iostream>
#include <vector>

#include "ms_utils.h"
#include "timer.hpp"
#include <chrono>

#include "mean_shift_postprocessing.h"

namespace MeanShift {
namespace seq {

namespace {
float _sq_dist( float const* const x , float const* const xi ,
				const unsigned int m ){
	float dist = 0;
	for( unsigned int j=0 ; j<m ; j++ )
		dist += (x[j] - xi[j])*(x[j] - xi[j]);
	return dist;
}

float _g_normal( float const* const x , float const* const xi,
				 const unsigned int m , const float h ){
	float g_val = _sq_dist(x , xi ,m)/h;
	if( g_val == 0 ) return 0;
	return std::exp( -0.5 * g_val );
}

void _ms_shift_p( float const* const p , float const* const data ,
				  float* p_next ,	// output
				  const unsigned long n, const unsigned int m ,
				  const float h ){

	float g_acc = 0;
	memset( p_next , 0 , m*sizeof(float) );
	for( unsigned long i=0 ; i < n ; i++ ){
		float g_val = _g_normal( p , &(data[i*m]) , m , h );
		g_acc += g_val;
		for( unsigned int j=0 ; j < m ; j++ )
			p_next[j] += data[i*m+j] * g_val;
	}
	for( unsigned int j=0 ; j < m ; j++ )
		p_next[j] /= g_acc;
}

bool _ms_convergence( float const* const msv , const unsigned int m ,
					  const float tol ){
	float sq_norm = 0;
	for( unsigned int j=0 ; j<m ; j++ )
		sq_norm += msv[j]*msv[j];
	return (sq_norm <= tol*tol);
}

void _ms_procedure( float const* const start , float const* const data ,
					float *convergence_p , // output, assumed allocated outside
				    const unsigned long n , const unsigned int m ,
				    const float h , const float tol,
				    const unsigned int max_iter ){

	float *cur, *next, *tmp;
	float *xb1 = new float[m];
	float *xb2 = new float[m];
	std::copy( start , &(start[m]) , xb1);

	float *msv = new float[m];

	unsigned int it = 0;
	cur = xb1;
	next = xb2;
	while( it==0 || ( it < max_iter && !_ms_convergence( msv , m , tol ) ) ){
		_ms_shift_p( cur , data , next , n , m , h );
		for( unsigned int j=0 ; j<m ; j++ )
			msv[j] = next[j] - cur[j];

		tmp = cur;
		cur = next;
		next = tmp;
		it++;
	}
	delete msv;
	delete next;
	std::copy( cur , cur+m , convergence_p );
	delete cur;
}
}

std::vector<Cluster_t> ms_clustering(
		MSResult* result ,
		float const* const starts , float const* const data ,
		const unsigned long n_start , const unsigned long n , const unsigned int m ,
		const float h, const float tol, const float agg_th , const unsigned int max_iter ){

	c8::Timer timer;

	// Compute convergence pts
	float* conv_pts = new float[n_start*m];
	timer.start();
	for( int i=0 ; i < n_start ; i++ )
		_ms_procedure( starts+i*m , data , conv_pts+i*m ,
					   n , m , h , tol , max_iter );
	timer.stop();
	result->mean_shift_time = timer.elapsed<std::chrono::nanoseconds>()/1000000.0;

	// Form clusters
	std::vector<Cluster_t> clusters;
	timer.start();
	clusters = _form_clusters_packed( conv_pts , n , m , agg_th );
	timer.stop();
	result->clusters_formation_time = timer.elapsed<std::chrono::nanoseconds>()/1000000.0;

	delete conv_pts;
	return clusters;
}

} // sequential namespace
} // MeanShift namespace
