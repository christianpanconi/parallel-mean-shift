#include "mean_shift_postprocessing.h"
#include <stdio.h>

/**
 * SQUARED DISTANCES FUNCTIONS
 */
//__host__
static float sq_dist_planar( float *data , unsigned int i1 , unsigned int i2 , unsigned int n, unsigned int m){
	if( i1 == i2 ) return 0;
	float sq_dist = 0;
	for( unsigned int j=0 ; j<m ; j++ )
		sq_dist += (data[j*n+i1]-data[j*n+i2])*(data[j*n+i1]-data[j*n+i2]);
	return sq_dist;
}

//__host__
static float sq_dist_packed( float *p1 , float *p2 , unsigned int m ){
	if( p1 == p2 ) return 0;
	float dist = 0;
	for( int j=0 ; j<m ; j++ )
		dist += (p1[j]-p2[j])*(p1[j]-p2[j]);
	return dist;
}

/**
 * Clusters formation functions.
 */

/**
 * Assumes planar layout for the convergence points.
 */
std::vector<Cluster_t> _form_clusters_planar(
	float *conv_pts , unsigned int n , unsigned int m ,
	float agg_th ){

	std::vector<Cluster_t> clusters;

	float sq_agg_th = agg_th*agg_th;
	for( unsigned int i=0 ; i<n ; i++ ){
		bool aggregated = false;
		unsigned int agg_i;

		for( unsigned int c=0 ; c<clusters.size() ; c++ ){
			for( unsigned int ci=0 ; ci < clusters[c].pts_indices.size() ; ci++ ){
				if( sq_dist_planar( conv_pts , i , clusters[c].pts_indices[ci] , n , m ) <= sq_agg_th ){
					agg_i = c;
					aggregated = true;
					break;
				}
			}
			if( aggregated ) break;
		}
		if( aggregated ){
			for( int j=0 ; j<m ; j++ ) // Update centroid
				clusters[agg_i].centroid[j] =
					(clusters[agg_i].centroid[j]*clusters[agg_i].pts_indices.size() + conv_pts[j*n+i]) /
						(clusters[agg_i].pts_indices.size()+1);
			clusters[agg_i].pts_indices.push_back( i );
		}else{
			Cluster_t new_cluster;
			new_cluster.centroid = new float[m];
			new_cluster.pts_indices.push_back(i);
			for( int j=0 ; j<m ; j++ )
				new_cluster.centroid[j] = conv_pts[j*n+i];
			clusters.push_back(new_cluster);
		}
	}

	// Merge clusters based on centroids distance
//	_merge_clusters( clusters , m , agg_th );
	return clusters;
}

/**
 * Assumes packed layout for the convergence points
 */
std::vector<Cluster_t> _form_clusters_packed(
	float* conv_pts , unsigned int n, unsigned int m, float agg_th ){

	std::vector<Cluster_t> clusters;

	float sq_agg_th = agg_th*agg_th;
	for( unsigned int i=0 ; i<n ; i++ ){
		bool aggregated = false;
		unsigned int agg_i;

		for( unsigned int c=0 ; c<clusters.size() ; c++ ){
			for( unsigned int ci=0 ; ci < clusters[c].pts_indices.size() ; ci++ ){
				if( sq_dist_packed( conv_pts+i*m , conv_pts+clusters[c].pts_indices[ci]*m , m ) <= sq_agg_th ){
					agg_i = c;
					aggregated = true;
					break;
				}
			}
			if( aggregated ) break;
		}
		if( aggregated ){
			for( int j=0 ; j<m ; j++ ) // Update centroid
				clusters[agg_i].centroid[j] =
					(clusters[agg_i].centroid[j]*clusters[agg_i].pts_indices.size() + conv_pts[i*m+j]) /
						(clusters[agg_i].pts_indices.size()+1);
			clusters[agg_i].pts_indices.push_back( i );
		}else{
			Cluster_t new_cluster;
			new_cluster.centroid = new float[m];
			new_cluster.pts_indices.push_back(i);
			for( int j=0 ; j<m ; j++ )
				new_cluster.centroid[j] = conv_pts[i*m+j];
			clusters.push_back(new_cluster);
		}
	}

	// Merge clusters based on centroids distance
//	_merge_clusters( clusters , m , agg_th );
	return clusters;
}
