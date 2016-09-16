# ##############################################################################
# Version 1: Initial version
# Author: Daniel Platz
# ##############################################################################
import os

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import my_paths2
import hdf5
import mpl_helpers

mpl_helpers.clear_all()


# ##############################################################################
# Parameters
# ##############################################################################

# Plotting
cmap = cm.coolwarm
dpi = 150
filename_fig1 = 'unit_cell_corsssection_transformation_3d.png'


# ##############################################################################
# Definitions
# ##############################################################################
def rotate_phases(phi0, isInverse=False):
    ''' Rotates the phases such that the phase space direction phi_0 is rotated
        onto the main diagonal of the n dimensional phase space

    	Author: Daniel Platz

    Parameters
    ----------
    phi  :  		np.array
    				array of phases
    isInverse  :  	bool
    			  	if True, rotates back the rotated phase space back to the physical phase space
    			  	(implies that isInverse=True gives you the coordinates in the rotated system)

    Returns
    -------
    phi_0_rotated  :  np.array
    				  phases in rotated or physical phase space '''
    # Determine rotation angle, depends on number of oscillators and hence on the dimension of the phase space of phases
    N = len(phi0)
    alpha = -np.arccos(1.0 / np.sqrt(N))

    # Construct rotation matrix
    v = np.zeros((N, N))
    v[0, 0] = 1.0
    v[1:, 1:] = 1.0 / float(N - 1)
    w = np.zeros((N, N))
    w[1:, 0] = -1.0 / np.sqrt(N - 1)
    w[0, 1:] = 1.0 / np.sqrt(N - 1)
    r = np.identity(N) + (np.cos(alpha) - 1) * v + np.sin(alpha) * w

    # Apply rotation matrix -- Notiz: check rounding errors in relation to modulus operations
    if not isInverse:
        return np.dot(r, phi0)
    else:
        return np.dot(np.transpose(r), phi0)


# ##############################################################################
# 1. Project 3D unit cell cross section
# ##############################################################################
print '1. Project 3D unit cell cross section'

# Parameters
n_dim = 3
n_points = 50

# Create points
x = np.zeros((n_points**(n_dim - 1), 3))
y = np.linspace(-np.pi, np.pi, n_points)
z = np.linspace(-np.pi, np.pi, n_points)
counter = 0
for iy in range(n_points):
    for iz in range(n_points):
        x[counter] = np.array([0, y[iy], z[iz]])
        counter += 1


# Points in rotated coordinates
x_rot = rotate_phases(x.transpose())
x_rot = x_rot.transpose()


fig = plt.figure(1, figsize=mpl_helpers.SIZE_PSI_SCREEN_HALF)

ax1 = fig.add_subplot(2, 1, 1, projection='3d')
#ax.set_aspect(1.0)
ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c='b', edgecolors='b', alpha=0.6, label='initial coordinates')
ax1.scatter(x_rot[:, 0], x_rot[:, 1], x_rot[:, 2], c='r', edgecolors='r', label='rotated coordinates')
ax1.set_xlabel('1. coordinate axis')
ax1.set_ylabel('2. coordinate axis')
ax1.set_zlabel('3. coordinate axis')
plt.legend(fontsize=9)

ax = fig.add_subplot(2, 1, 2)
ax.set_aspect('equal')
ax.scatter(x[:, 1], x[:, 2], c='b', edgecolors='b', alpha=0.6, label='initial coordinates')
ax.scatter(x_rot[:, 1], x_rot[:, 2], c='r', edgecolors='r', label='rotated coordinates')
plt.xlabel('2. coordinate axis')
plt.ylabel('3. coordinate axis')
plt.legend(fontsize=9)



# ##############################################################################
# Store data
# ##############################################################################
print 'Store data'

# Figures
#mpl_helpers.grid_all()
mpl_helpers.tight_layout_and_grid_all()
plt.figure(1)
plt.savefig(filename_fig1, dpi=dpi)
plt.show()