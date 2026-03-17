#!/bin/bash

echo "Running baseline (non-MPI)..."
baseline=$(python mandelbrot_vec.py) 

echo "cores,time" > results.csv

for p in {1..16}
do
    echo "Benchmarking $p processors..."
    mpiexec -n $p python mandelbrot_vec_parallel.py >> results.csv || exit 1
done

echo "displaying speedup_graph.."
python scaling_graphs.py $baseline