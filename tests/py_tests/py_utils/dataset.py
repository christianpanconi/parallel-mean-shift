import numpy as np
from PIL import Image
import math

# ================
# DATASET UTILS
# ================
def load_image_dataset( img_path , normalize_rgb=True , normalize_pos=True ):
    """
    Loads the given image with dimension WxH as a numpy array with shape (W*H,5),
    where each elements contains [ R , G , B , x , y ].
    The features will be normalized in the [0,1] range if specified (True by default).
    """
    img = Image.open(img_path)
    print( "{0} / {1} / {2} | {3} pixels".format(
        img.format , img.size , img.mode , img.size[0]*img.size[1] ) )

    # np_img[0][0] is top left
    # first index  -> row index
    # second index -> column index
    np_img = np.asarray(img)
    if normalize_rgb:
        np_img = np_img / 255.0
    row_max = img.size[0]-1
    col_max = img.size[0]-1

    # (i,j) -> l := i*img.size[0]+j
    # i = floor( l / img.size[0] )
    # j = l % img.size[0]
    dataset = []
    for i in range(0,img.size[0]):
        for j in range(0,img.size[1]):
            dataset.append( [
                np_img[i][j][0] , # R
                np_img[i][j][1] , # G
                np_img[i][j][2] , # B
                i/row_max if normalize_pos else i , # x
                j/col_max if normalize_pos else j   # y
            ] )
    del np_img
    dataset = np.array(dataset)
    return dataset , (img.size[0],img.size[1])

def normalize_dataset( dataset: list ):
    """
    Rescale the dataset elements features to the [0,1] range.
    Returns the rescaled dataset along with a list of tuples containing
    (min,max) for each dimension.
    """
    dim_bounds = []
    for i in range(dataset[0].shape[0]):
        dim_bounds.append((
            np.min([d[i] for d in dataset]) ,
            np.max([d[i] for d in dataset])
        ))
    norm_dataset = []
    for d in dataset:
        norm_dataset.append( np.array([(d[i]-dim_bounds[i][0])/(dim_bounds[i][1]-dim_bounds[i][0]) for i in range(len(dim_bounds))]) )
    return np.array(norm_dataset), dim_bounds

def sort_blocks_image_dataset( dataset , img_w , img_h , block_size ):
    nbx = math.ceil(img_w / block_size)
    nby = math.ceil(img_h / block_size)

    blocks = [[] for i in range(nbx*nby)]
    indices = [[] for i in range(nbx * nby)]

    for i in range(len(dataset)):
        r = int( math.floor(i/img_w) )
        c = i % img_w
        br = int(math.floor(r/block_size))
        bc = int(math.floor(c/block_size))
        blocks[br*nbx + bc].append( dataset[i] )
        indices[br*nbx + bc].append( (r,c) )

    for i in range(1,len(blocks)):
        blocks[0].extend(blocks[i])
        indices[0].extend(indices[i])

    indices = indices[0]
    blocks_dataset = np.asarray(blocks[0])
    # indices maps the new index of each pixel in the new order
    # to the row and column in the original image
    return blocks_dataset, indices