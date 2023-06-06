# This python program is an image segmentation example
# using the mean_shift module.

import mean_shift as ms
from py_utils.dataset import load_image_dataset, sort_blocks_image_dataset
import numpy as np
import datetime
from PIL import Image

# Load image
IMAGE_PATH = "../images/urban2/256/test.jpg"
IMAGE_NAME = IMAGE_PATH.split("/")[-1]
IMAGE_NAME = IMAGE_NAME[:IMAGE_NAME.rfind(".")]
dataset, image_size = load_image_dataset( IMAGE_PATH )

# MS params
H = 0.005
TOL = 0.001
AGG_TH = 0.02
MAX_ITER = 100
KERNEL_TYPE = "simple"
BLOCK_SIZE = 64

SORTING_BLOCK = False
SORTING_BLOCK_SIZE = 8

if SORTING_BLOCK:
    dataset, indices = sort_blocks_image_dataset( dataset ,
        image_size[0] , image_size[1] , SORTING_BLOCK_SIZE )

# Mean Shift Clustering
print( "Performing Mean Shift Clustering ..." )
print( "Started at {0}".format( datetime.datetime.now() ) )

# Sequential
# result = ms.ms_clustering(
#     dataset , starts=dataset ,
#     h=H , tol=TOL , agg_th=AGG_TH ,
#     max_iter=MAX_ITER
# )

# CUDA
result = ms.cu_ms_clustering(
    dataset ,
    h=H , tol=TOL , agg_th=AGG_TH ,
    max_iter=MAX_ITER , kernel_type=KERNEL_TYPE ,
    block_size=BLOCK_SIZE )

print( "MS time: {0:.3f} ms".format( result["mean_shift_time"] ) )
print( "Clusters formation time: {0:.3f} ms".format( result["clusters_formation_time"] ) )

# Build the segmented image from the clusters
segmented = np.zeros( (image_size[0] , image_size[1] , 3) )
colors = [ [int(c[0]*255) , int(c[1]*255) , int(c[2]*255)]
           for c in result["centroids"] ]
for ci,cluster in enumerate(result["clusters"]):
    for p in cluster:
        if not SORTING_BLOCK:
            i = int(p/image_size[0])
            j = int(p%image_size[0])
        else:
            i , j = indices[p]
        segmented[i][j][0] = colors[ci][0]
        segmented[i][j][1] = colors[ci][1]
        segmented[i][j][2] = colors[ci][2]

# Save segmented image to file
segmented_img = Image.fromarray( segmented.astype(np.uint8) )
segmented_img.save( IMAGE_NAME + "_segmented.jpg" )
print( "Segmented image saved as: {0}_segmented.jpg".format(IMAGE_NAME) )
