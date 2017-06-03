# ##############################################################################
# Plot the different coupling functions provided by synctools
# Author: Daniel Platz
# ##############################################################################
import os

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import synctools


# ##############################################################################
# Parameters
# ##############################################################################

# Plotting
cmap = cm.coolwarm
dpi = 150


# ##############################################################################
#
# ##############################################################################
print ''


freq = 1.0 / (2 * np.pi)
phi = np.linspace(-3 * np.pi, 3 * np.pi, 1000)

sin = synctools.Sin(freq)
dsindt = sin.get_derivative()

plt.figure(1)
gs = plt.GridSpec(2, 1)
plt.subplot(gs[0])
plt.plot(phi, sin(phi), 'b-', lw=2)

plt.subplot(gs[1])
plt.plot(phi, dsindt(phi), 'b-', lw=2)


tri = synctools.Triangle(freq)
dtridt = tri.get_derivative()


plt.figure(2)
gs = plt.GridSpec(2, 1)
plt.subplot(gs[0])
plt.plot(phi, tri(phi), 'b-', lw=2)

plt.subplot(gs[1])
plt.plot(phi, dtridt(phi), 'b-', lw=2)




# ##############################################################################
# Store data
# ##############################################################################
print 'Store data'

# Figures
#mpl_helpers.grid_all()
#mpl_helpers.tight_layout_and_grid_all()
plt.show()