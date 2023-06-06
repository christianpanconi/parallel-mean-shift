import mean_shift as ms
from py_utils.dataset import load_image_dataset
from py_utils.dataset import sort_blocks_image_dataset

import json
import numpy as np
import time
from math import ceil, sqrt
from pathlib import Path
from prettytable import PrettyTable

import datetime

import argparse
parser = argparse.ArgumentParser(
    prog="python3 ms_clustering_benchmark.py",
    description="Benchmark for the CUDA Mean Shift."
)
# BENCHMARK args
parser.add_argument( "-i" , "--image" )
parser.add_argument( "-d" , "--dataset" )
parser.add_argument( "-tf" , "--test-folder" )
parser.add_argument( "-bmkit" , "--benchmark-iter", type=int , default=10 )
parser.add_argument( "-s" , "--sequential" , action="store_true" )
parser.add_argument( "-n" , "--no-out" , action="store_true" )
parser.add_argument( "--sorting" , default="raster" , choices=["raster","shuffle","blocks"] )
parser.add_argument( "--sorting-block-size" , type=int , default=8 )
# MS params args
parser.add_argument( "-msh" , "--meanshift-h" , type=float, default=0.005 )
parser.add_argument( "-mstol" , "--meanshift-tol" , type=float , default=0.001 )
parser.add_argument( "-msaggth" , "--meanshift-agg-th" , type=float , default=0.02 )
parser.add_argument( "-msmaxit" , "--meanshift-max-iter" , type=int , default=100 )
# CUDA specific args
parser.add_argument( "-cudakt" , "--cuda-kernel-type" , default="simple" ,
                     choices=["simple","tiled"] )
parser.add_argument( "-cudabs" , "--cuda-block-size" , type=int , default=32 )
args = parser.parse_args()

if args.test_folder is None:
    TEST_FOLDER = "./"
else:
    TEST_FOLDER = args.test_folder + ("/" if not args.test_folder.endswith("/") else "")

TEST_FOLDER = "./" if args.test_folder is None else args.test_folder
TEST_FOLDER += "/" if not TEST_FOLDER.endswith("/") else ""
if args.image is None and args.dataset is None:
    print( "Neither --image nor --dataset are specified. Exiting." )
    exit(0)
elif args.image is not None and args.dataset is not None:
    print( "Both --image and --dataset are specified. Specify only one of them. Exiting." )
    exit(0)
else:
    data_path = args.image if args.image is not None else args.dataset
data_path = TEST_FOLDER + data_path

# BENCHMARK PARAMETERS
BMK_ITS = args.benchmark_iter
SEQ = args.sequential
SORTING = args.sorting
if SORTING == "blocks":
    SORTING_BLOCK_SIZE = args.sorting_block_size
# MEAN SHIFT PARAMETERS
H = args.meanshift_h 
TOL = args.meanshift_tol 
AGG_TH = args.meanshift_agg_th 
MAX_ITER = args.meanshift_max_iter
# CUDA KERNEL PARAMETERS
KERNEL_TYPE = args.cuda_kernel_type
BLOCK_SIZE = args.cuda_block_size

# Load dataset
if args.image is not None:
    print("\nImage: " + data_path)
    dataset, img_size = load_image_dataset(data_path)
else:
    print( "\nDataset: " + data_path )
    with open(data_path) as data_file:
        dataset = np.asarray( json.load(data_file) )
print("Dataset shape: {0}".format(dataset.shape))
# Kernel info:
if not SEQ:
    print("Block Size: ({0},1,1)\t|\tGrid Size: ({1},1,1)\n".format(
        BLOCK_SIZE, int(ceil((dataset.shape[0]) / BLOCK_SIZE))
    ))
    kernel_info = ms.cu_ms_kernel_info(kernel_type=KERNEL_TYPE,n_dims=5,block_size=BLOCK_SIZE)
    kernel_info_table = PrettyTable()
    kernel_info_table.field_names = ["Kernel launch property" , "Value" ]
    kernel_info_table.align["Value"] = 'r'
    kernel_info_table.add_rows([ [k,v] for k,v in kernel_info.items()] )
    print( "Kernel type: {0}".format(KERNEL_TYPE) )
    print(kernel_info_table)
print("\n")

# ----------------
# Prepare Mean shift clustering call
if SEQ:
    def mean_shift_clustering():
        return ms.ms_clustering(dataset, h=H, tol=TOL, max_iter=MAX_ITER, agg_th=AGG_TH)
else:
    def mean_shift_clustering():
        return ms.cu_ms_clustering( dataset, h=H , tol=TOL, max_iter=MAX_ITER, agg_th=AGG_TH,
                                    kernel_type=KERNEL_TYPE, block_size=BLOCK_SIZE )
# ----------------

# Blocks sorting
if SORTING == "blocks":
    dataset , indices = sort_blocks_image_dataset(dataset , img_size[0] , img_size[1] , block_size=SORTING_BLOCK_SIZE )

print( "Started at {0}".format(datetime.datetime.now()) )

# MS Benchmark runs:
results = []
for i in range(BMK_ITS):
    print( "Benchmark run: {0} / {1}".format(i+1,BMK_ITS).ljust(40), end="")
    if SORTING == "shuffle":
        np.random.shuffle(dataset)
    result = mean_shift_clustering()
    results.append({
        "mean_shift_time": result["mean_shift_time"],
        "clusters_formation_time": result["clusters_formation_time"],
        "n_clusters": len(result["clusters"])
    })
    print( "| MS: {0:.3f} ms".format(result['mean_shift_time']).ljust(30),end="" )
    print( "| clustering: {0:.3f} ms".format(result['clusters_formation_time']).ljust(30) )

# Metrics (AVG exec time and std)
avg_ms_time = avg_cl_time = 0
for r in results:
    avg_ms_time += r["mean_shift_time"]
    avg_cl_time += r["clusters_formation_time"]
avg_ms_time /= len(results)
avg_cl_time /= len(results)
ms_std = cl_std = 0
for r in results:
    ms_std += (avg_ms_time-r["mean_shift_time"])**2
    cl_std += (avg_cl_time-r["clusters_formation_time"])**2
ms_std = sqrt(ms_std / len(results))
cl_std = sqrt(cl_std / len(result))

print( "-"*40 )
print( "AVG ms time: {0:.3f} ms".format( avg_ms_time ) )
print( "ms time STD: {0:.3f} ms / {1:.2f} %".format(ms_std , (ms_std/avg_ms_time)*100 ) )
print( "AVG cl time: {0:.3f} ms".format( avg_cl_time ) )
print( "cl time STD: {0:.3f} ms / {1:.2f} %".format( cl_std , (cl_std/avg_cl_time)*100 ) )
print( "-"*40 )

# Save benchmark result to file
if not args.no_out:
    bmk_result = {
        "bmk_it": BMK_ITS ,
        "sorting": SORTING ,
        "results": results ,
        "ms_params": { "h": H, "tol": TOL, "agg_th": AGG_TH , "max_iter": MAX_ITER } ,
        "avg_ms_time": avg_ms_time,
        "ms_time_std": ms_std,
        "ms_time_std_perc": (ms_std/avg_ms_time),
        "avg_cl_time": avg_cl_time,
        "cl_time_std": cl_std,
        "cl_time_std_perc": (cl_std/avg_cl_time)
    }
    if SORTING == "blocks":
        bmk_result["sorting_blocks_size"] = SORTING_BLOCK_SIZE
    if not SEQ:
        bmk_result["kernel_type"] = KERNEL_TYPE
        bmk_result["kernel_launch_info"] = kernel_info
    OUT_FOLDER = TEST_FOLDER + "benchmarks"
    Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)
    OUT_FILE = OUT_FOLDER + "/bmk_" + ("sequential" if SEQ else (KERNEL_TYPE+str(BLOCK_SIZE))) + "_" + str(int(time.time_ns()/10**6)) + ".json"
    with open(OUT_FILE , "w") as f:
        f.write( json.dumps(bmk_result, indent=4) )
    print( "Benchmark result saved as: '{0}'".format(OUT_FILE) )
