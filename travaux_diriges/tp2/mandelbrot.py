# Calcul de l'ensemble de Mandelbrot en python
from dataclasses import dataclass
from math import log
from time import time

import matplotlib.cm
import numpy as np
from mpi4py import MPI
from PIL import Image

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        z: complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.0j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.0e-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z))) / log(2)
                return iter
        return self.max_iterations


# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=2.0)

globCom = MPI.COMM_WORLD.Dup()
nbp = globCom.size
rank = globCom.rank

width, height = 1024, 1024
parcel_size = height // nbp
height = parcel_size * nbp

scaleX = 3.0 / width
scaleY = 2.25 / height

deb = time()

convergence = np.empty((width * parcel_size), dtype=np.double)

# Calcul de l'ensemble de mandelbrot :
for y in range(parcel_size):
    for x in range(width):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * (rank * parcel_size + y))
        convergence[y * width + x] = mandelbrot_set.convergence(c, smooth=True)



fin = time()
print(f"Temps du calcul de l'ensemble de Mandelbrot : {fin-deb}")

glob_array = None
if rank==0: glob_array = np.empty(width * height,dtype=np.double)

globCom.Gather(convergence, glob_array)

if rank == 0:
    # Constitution de l'image résultante :
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(glob_array.T) * 255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()
