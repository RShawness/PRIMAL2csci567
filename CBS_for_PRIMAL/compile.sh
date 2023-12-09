#!/bin/bash

# Check if pybind11 directory exists
if [ ! -d "pybind11" ]; then
  # If not, download it from GitHub
  git clone https://github.com/pybind/pybind11.git
fi

mkdir -p build

# build exec for cpp

cd build
cmake ../ -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
make -j


# build exec for python

# cd build
# cmake ../ -DPYTHON=true
# make -j
