from time import time

import numpy as np
from mpi4py import MPI
import os


class Bucket:
    buff: np.array

    def __init__(self, buffer):
        self.buff = buffer

    def sort(self):
        return np.sort(self.buff)

    def quantiles(self, n_quantiles):
        q = []
        for i in range(n_quantiles - 1):
            q.append(np.quantile(self.buff, (i + 1) / n_quantiles))
        return q

    def cut(self, quantile_list):
        c = np.digitize(self.buff, quantile_list)
        e = [[] for _ in range(len(quantile_list) + 1)]
        for i in range(len(self.buff)):
            e[c[i]].append(self.buff[i])
        return e


def init_random_list(n_elem):
    return np.random.rand(n_elem)


def inhomogenous_flatten(L):
    if L == None:
        return L
    return [item for sublist in L for item in sublist]


def all_equal(a, b):
    for ea, eb in zip(a, b):
        if ea != eb:
            return False
    return True


def make_data(nproc, rank, length):
    length -= length % nproc
    if rank == 0:
        raw_data = init_random_list(length)
        data = np.reshape(raw_data, (nb_p, (int)(LENGTH / nb_p)))
    else:
        data = None
    return data


if __name__ == "__main__":

    LENGTH = 1048576  # 2^20 -- for a 2^i  (i < 20) number of processes

    root = 0

    comm = MPI.COMM_WORLD
    nb_p = comm.Get_size()
    rank = comm.Get_rank()

    MPI.COMM_WORLD.barrier()

    data = make_data(nb_p,rank,LENGTH)
    b = Bucket(comm.scatter(data, root))

    if rank == 0:
        deb = time()

    b.sort()

    split_points = b.quantiles(nb_p)
    macro_splits = Bucket(np.array(comm.allgather(split_points)).flatten())
    real_splits = macro_splits.quantiles(nb_p)

    data_to_send = b.cut(real_splits)
    receive_bucket = inhomogenous_flatten(comm.alltoall(data_to_send))
    receive_bucket = np.sort(receive_bucket)

    MPI.COMM_WORLD.barrier()

    affinity = os.sched_getaffinity(0)
    print(f"Rank {rank} is pinned to Core(s): {list(affinity)}")

    if rank == 0:
        fin = time()
        print(f"{nb_p},{fin-deb}")

        # final_result = inhomogenous_flatten(comm.gather(receive_bucket, root))

    # execute with : mpiexec -np 2 python bucket.py
