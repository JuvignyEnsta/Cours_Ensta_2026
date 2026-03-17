echo "cores,time" > results.csv
for p in {1..16}
do
    echo "working with $p processors"
    mpiexec -n $p python bucket_optimize.py >> results.csv || exit 1
done