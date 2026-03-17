from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from time import *

@jit(nopython=True, parallel=True)
def compute_mandelbrot(h, w, max_iter):
    # Pre-allocate the image array
    img = np.zeros((h, w), dtype=np.uint32)
    
    for i in range(h):
        # Setting up the complex plane
        cy = -1.5 + (i * 3.0 / h)
        for j in range(w):
            cx = -2.0 + (j * 3.0 / w)
            x, y = 0.0, 0.0
            iteration = 0
            while x*x + y*y <= 4 and iteration < max_iter:
                x_new = x*x - y*y + cx
                y = 2*x*y + cy
                x = x_new
                iteration += 1
            img[i, j] = iteration
    return img

# This will feel "instant" compared to standard Python
deb = time()
result = compute_mandelbrot(1000, 1000, 100)
print(time() - deb)
plt.imshow(result, cmap='magma')
plt.show()