import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Parameters
n = 1000
x_max = 8 * np.pi

# Other quantities
dx = x_max / n
dy = dx
z = np.zeros((n, n))

# Create z values
for i in range(n):
    for j in range(n):
        z[i, j] = np.sin(i * dy) * np.sin(j * dx)

# Create masked array
z_ma = ma.masked_where(z < 0, z)

# Figure 1: Plot all data
cmap = cm.coolwarm
plt.figure(1)
plt.clf()
plt.imshow(z, aspect='auto', interpolation='None', cmap=cmap)
plt.colorbar()

# Figure 3: Plot masked array with custom mask color
cmap.set_bad('k')
plt.figure(3)
plt.clf()
plt.imshow(z_ma, aspect='auto', interpolation='None', cmap=cmap)
plt.colorbar()
plt.draw()