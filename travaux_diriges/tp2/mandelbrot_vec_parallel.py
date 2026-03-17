# Calcul de l'ensemble de Mandelbrot en python
from dataclasses import dataclass
from math import log
from time import time

import matplotlib.cm
import numpy as np
from mpi4py import MPI
from PIL import Image


class MandelbrotSet:

    def __init__(self, max_iterations: int, escape_radius: float = 2.0):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: np.ndarray, smooth=False, clamp=True) -> np.ndarray:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return np.maximum(0.0, np.minimum(value, 1.0)) if clamp else value

    def count_iterations(self, c: np.ndarray, smooth=False) -> np.ndarray:
        z: np.ndarray
        iter: np.ndarray

        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        iter = self.max_iterations * np.ones(c.shape, dtype=np.double)
        mask = (np.abs(c) >= 0.25) | (np.abs(c + 1.0) >= 0.25)
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}

        z = np.zeros(c.shape, dtype=np.complex128)
        for it in range(self.max_iterations):
            z[mask] = z[mask] * z[mask] + c[mask]
            has_diverged = np.abs(z) > self.escape_radius
            if has_diverged.size > 0:
                iter[has_diverged] = np.minimum(iter[has_diverged], it)
                mask = mask & ~has_diverged
            if np.any(mask) == False:
                break
        has_diverged = np.abs(z) > 2
        if smooth:
            iter[has_diverged] += 1 - np.log(np.log(np.abs(z[has_diverged]))) / log(2)
        return iter


# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=200, escape_radius=2.0)

globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank = globCom.rank

width, height = 2048, 2048
parcel_size = height // nbp
height = parcel_size * nbp

scaleX = 3.0 / width
scaleY = 3.0 / height

convergence = np.empty((parcel_size,width), dtype=np.double)

begin_x = -2.0
begin_y = -1.5 + scaleY * (rank * parcel_size)

x = begin_x + scaleX * np.arange(width)
y = begin_y + scaleY * np.arange(parcel_size)

X, Y = np.meshgrid(x, y)
C = X + 1j * Y

globCom.Barrier()
deb = MPI.Wtime()
convergence = mandelbrot_set.convergence(C, smooth=True)

globCom.Barrier()
fin = MPI.Wtime()
# print(f"Vectorized calculation for processor rank {rank} took: {fin - deb:.4f}s")
if rank == 0: print (f"{nbp},{fin-deb : .2f}")

full_convergence = None
if rank == 0:
    full_convergence = np.empty((height, width), dtype=np.double)

globCom.Gather(convergence, full_convergence, root=0)

if rank == 0:
    # print(f"Total image gathered. Shape: {full_convergence.shape}")
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.afmhot(full_convergence) * 255))
    fin = time()
    # print(f"Temps de constitution de l'image : {fin-deb}")
    # image.save("mandel.png")
