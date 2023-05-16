/**
 * Mean shift custom lib
 */
#ifndef LIB_MEAN_SHIFT_INCLUDE_MEAN_SHIFT_H_
#define LIB_MEAN_SHIFT_INCLUDE_MEAN_SHIFT_H_

#include <vector>

#include "../mean_shift_postprocessing.h"

//float _sq_dist( float const* const x , float const* const xi ,
//				const unsigned long m );
//
//float _g_normal( float const* const x , float const* const xi,
//				 const unsigned int m , const float h );
//
//void _ms_shift_p( float const* const p , float const* const data ,
//				  float* p_next ,
//				  const unsigned long n, const unsigned int m ,
//				  const float h );
//
//bool _ms_convergence( float const* const msv , const unsigned int m ,
//					  const float tol );
//
//void _ms_procedure( float const* const start , float const* const data ,
//					float * convergence_p ,
//				    const unsigned long n , const unsigned int m ,
//				    const float h , const float tol,
//				    const unsigned int max_iter );

namespace MeanShift {
namespace seq{

std::vector<Cluster_t> ms_clustering(
		MSResult* result ,
		float const* const starts , float const* const data ,
		const unsigned long n_start , const unsigned long n , const unsigned int m ,
		const float h, const float tol, const float agg_th , const unsigned int max_iter);


} // sequential namespace
} // MeanShift namespace
#endif /* LIB_MEAN_SHIFT_INCLUDE_MEAN_SHIFT_H_ */
