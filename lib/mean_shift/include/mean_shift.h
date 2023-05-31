/**
 * Mean shift custom lib
 */
#ifndef LIB_MEAN_SHIFT_INCLUDE_MEAN_SHIFT_H_
#define LIB_MEAN_SHIFT_INCLUDE_MEAN_SHIFT_H_

#include <vector>

#include "../mean_shift_postprocessing.h"

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
