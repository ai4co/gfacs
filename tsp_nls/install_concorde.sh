#!/bin/bash
mkdir concorde
cd concorde
mkdir qsopt
cd qsopt
# Download qsopt
if [[ "$OSTYPE" == "darwin"* ]]; then
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt.a
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt.h
    curl -O http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/mac64/qsopt
else
    wget https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.a
    wget https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.h
    wget https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt
fi
cd ..
wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar xf co031219.tgz
cd concorde
if [[ "$OSTYPE" == "darwin"* ]]; then
    ./configure --with-qsopt=$(pwd)/../qsopt --host=powerpc-apple-macos
else
    ./configure --with-qsopt=$(realpath ../qsopt)
fi
make
TSP/concorde -s 99 -k 100
cd ../..