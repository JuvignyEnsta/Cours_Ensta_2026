import os
from time import time

import numpy as np
from mpi4py import MPI


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
        return [self.buff[c == i] for i in range(len(quantile_list) + 1)]


def init_random_list(n_elem):
    return np.random.rand(n_elem)


def inhomogenous_flatten(L):
    if L == None or len(L) == 0:
        return L
    return np.concatenate(L)


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

    LENGTH = 10**8

    root = 0

    comm = MPI.COMM_WORLD
    nb_p = comm.Get_size()
    rank = comm.Get_rank()

    MPI.COMM_WORLD.barrier()

    data = make_data(nb_p, rank, LENGTH)
    b = Bucket(comm.scatter(data, root))

    if rank == 0:
        deb = time()

    #b.sort()

    split_points = b.quantiles(nb_p)
    macro_splits = Bucket(np.array(comm.allgather(split_points)).flatten())
    real_splits = macro_splits.quantiles(nb_p)

    data_to_send = b.cut(real_splits)
    receive_bucket = inhomogenous_flatten(comm.alltoall(data_to_send))
    receive_bucket = np.sort(receive_bucket)

    MPI.COMM_WORLD.barrier() #the program has only finished once everyone is done

    if rank == 0:
        exec_time = time() - deb

    affinity = os.sched_getaffinity(0)
    cores = comm.gather(affinity, root)

    if rank == 0:
        print(nb_p, exec_time, cores)
        begin = time() 
        np.random.rand(LENGTH).sort()
        end = time()
        # print(f"Single processor sort time: {end - begin} seconds")

        # final_result = inhomogenous_flatten(comm.gather(receive_bucket, root))

    # execute with : mpiexec -np 2 python bucket.py
