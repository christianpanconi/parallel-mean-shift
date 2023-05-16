#ifndef LIB_MEAN_SHIFT_MEAN_SHIFT_POSTPROCESSING_H_
#define LIB_MEAN_SHIFT_MEAN_SHIFT_POSTPROCESSING_H_

#include <vector>

typedef struct MSResult{
	float mean_shift_time;
	float clusters_formation_time;
} MSResult;

typedef struct Cluster_t{
	std::vector<unsigned int> pts_indices;
	float *centroid;
} Cluster_t;

std::vector<Cluster_t> _form_clusters_planar(
	float *conv_pts , unsigned int n , unsigned int m ,
	float agg_th );

std::vector<Cluster_t> _form_clusters_packed(
	float* conv_pts , unsigned int n, unsigned int m, float agg_th );

#endif /* LIB_MEAN_SHIFT_MEAN_SHIFT_POSTPROCESSING_H_ */
