# Parallel Mean Shift
Parallel [Mean Shift](https://en.wikipedia.org/wiki/Mean_shift) using **CUDA**.

This project includes:
* C++ library
* python package

## Dependecies
The C++ library has no external dependecies other than `CUDA`.

The python package requires `NumPy` to be present among the installed python packages.

## Build/Install
`CMake` version >= `3.18` is required.\
The following instructions assumes this repository cloned in a folder named `parallel-mean-shift`.

### C++ library
```bash
mkdir mean-shift/build
cd parallel-mean-shift/build
cmake .. -DCMAKE_BUILD_TYPE:String=Release \
  -DCMAKE_CUDA_ARCHITECTURES=<target compute capabilities>
make all
```

The `<target compute capabilities>` must be the CC identifier, for example to compile for CC 6.1 use `-DCMAKE_CUDA_ARCHITECTURES=61`.

The library will be located in `mean-shift/build/lib`.

Once the library has been successfully built, it can be installed with:
```bash
make install
```

### Python package
The python package can be built using the `setup.py` script:
```bash
cd parallel-mean-shift
python3 setup.py build --build-type=Release --cuda-archs=<target compute capabilities>
```

Once the package has been successfully built, it can be installed using `pip`:
```bash
pip install .
```

The package is named `mean-shift` and can be uninstalled with:
```bash
pip uninstall mean-shift
```
