import numpy as np

x = np.array([-2, -1, 0])[np.newaxis,:]
y = np.array([10, 20])[:,np.newaxis]

print(x)
print(y)
# Turn x into a "row" (1, 3) and y into a "column" (2, 1)
grid = x[np.newaxis, :] + y[:, np.newaxis]

print(grid)