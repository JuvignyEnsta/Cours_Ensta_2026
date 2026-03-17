import os
from time import time
import numpy as np
from mpi4py import MPI

class Bucket:
    def __init__(self, buffer):
        self.buff = np.ascontiguousarray(buffer, dtype=np.float64)

    def quantiles(self, n_quantiles):
        probs = np.linspace(0, 1, n_quantiles + 1)[1:-1]
        return np.quantile(self.buff, probs)

    def cut(self, quantile_list):
        indices = np.digitize(self.buff, quantile_list)
        return [self.buff[indices == i] for i in range(len(quantile_list) + 1)]

if __name__ == "__main__":
    LENGTH = 10**7
    comm = MPI.COMM_WORLD
    nb_p = comm.Get_size()
    rank = comm.Get_rank()
    root = 0

    LENGTH -= LENGTH % nb_p

    # 0. Distribute Data
    if rank == root:
        raw_data = np.random.rand(LENGTH).astype(np.float64)
        data_to_scatter = np.split(raw_data, nb_p)
    else:
        data_to_scatter = None

    local_data = comm.scatter(data_to_scatter, root)
    b = Bucket(local_data)

    comm.Barrier()
    if rank == root: deb = time()
    if nb_p > 1:
        # 1. Trouver les quantiles de son bucket
        local_splits = b.quantiles(nb_p).astype(np.float64)
        all_splits = np.empty(nb_p * (nb_p - 1), dtype=np.float64)
        # 2. Partager les quantiles avec les autres
        comm.Allgather(local_splits, all_splits) 
        
        # 3. Calculer les macro-quantiles globaux
        macro_splits = Bucket(all_splits)
        real_splits = macro_splits.quantiles(nb_p)

        # 4. Decouper son bucket selon ces limites
        data_to_send = b.cut(real_splits)

        # 5. Partager les Decoupes
        # Calculate how much we are sending to each rank
        send_counts = np.array([len(part) for part in data_to_send], dtype=np.int32)
        recv_counts = np.zeros(nb_p, dtype=np.int32)
        
        # Share the counts so everyone knows what to expect
        comm.Alltoall(send_counts, recv_counts)

        # Calculate displacements (offsets in memory)
        send_displ = np.insert(np.cumsum(send_counts[:-1]), 0, 0)
        recv_displ = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)

        # Flatten local data for buffered send
        flat_send_data = np.concatenate(data_to_send).astype(np.float64)
        receive_buffer = np.empty(sum(recv_counts), dtype=np.float64)

        comm.Alltoallv(
            [flat_send_data, send_counts, send_displ, MPI.DOUBLE],
            [receive_buffer, recv_counts, recv_displ, MPI.DOUBLE]
        )
    
    else : receive_buffer = b.buff

    # 6. Trier les donnes recues localement
    receive_buffer.sort()

    comm.Barrier()
    if rank == root:
        exec_time = time() - deb
        print(f"{nb_p},{exec_time:.4f}")
        
        # Validation compare
        begin = time() 
        np.sort(np.random.rand(LENGTH))
        print(f"Single Processor: {time() - begin:.4f}s")