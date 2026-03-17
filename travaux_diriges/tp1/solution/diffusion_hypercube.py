# Diffusion d'un jeton dans un hyper cube de dimension log_{2}(N)
from math import log2
from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup()
rank = globCom.rank
nbp  = globCom.size

dim = int(log2(nbp))
assert(2**dim == nbp), "Le nombre de processus doit être une puissance de 2"
if rank==0:
    print (f"Dimension du cube : {dim}")
jeton = None
if rank == 0:
    jeton = 42  # Le jeton initial

for d in range(dim):
    if rank < 2**d:
        # Processus émetteur
        dest = rank + 2**d
        print(f"Processus {rank} envoie le jeton {jeton} au processus {dest}",flush=True)
        globCom.send(jeton, dest=dest, tag=d)
    elif rank < 2**(d+1):
        # Processus récepteur
        src = rank - 2**d
        jeton = globCom.recv(source=src, tag=d)
        print(f"Processus {rank} a reçu le jeton {jeton} du processus {src}",flush=True)
print(f"Processus {rank} termine avec le jeton {jeton}")