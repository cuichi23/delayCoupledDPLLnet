# ##############################################################################
# Plot the parametric synchronized frequency curves for different coupling topologies
# Author: Daniel Platz
# Version 1: Inittial version
# ##############################################################################
import os

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import synctools
reload(synctools)



# ##############################################################################
# Parameters
# ##############################################################################

# Parameters
w = (2 * np.pi) * 1.00
k = (2 * np.pi) * 0.25
h = synctools.Triangle(1.0 / (2 * np.pi))
n = 12
nx = 4
ny = 3
topo_ring = synctools.Ring(n)
topo_square_periodic = synctools.SquarePeriodic(nx, ny)
twist_number1d = synctools.Twist1D(1)
twist_number2d = synctools.Twist2D(1, 0)

# Plotting
cmap = cm.coolwarm
dpi = 150



# ##############################################################################
# 1. Parametric m=1 twist frequency curves for different topologies for 12 coupled osci
# ##############################################################################
print '1. Parametric m=1 twist frequency curves for different topologies for 12 coupled osci'


# Parametric frequency curves
s_min =  0.0
s_max = 20.0
ds = 0.01
tau_ring, omega_ring, s_ring = synctools.get_parametric_omega_curve2(topo_ring, twist_number1d, h, k, w, s_min, s_max, ds)
tau_square_periodic, omega_square_periodic, s_square_periodic = synctools.get_parametric_omega_curve2(topo_square_periodic, twist_number2d, h, k, w, s_min, s_max, ds)


# Explicit
tau = np.linspace(0, 2, 500)
tau2_ring, omega2_ring = synctools.get_omega_curve(topo_ring, twist_number1d, h, k, w, tau)
tau2_square_periodic, omega2_square_periodic = synctools.get_omega_curve(topo_square_periodic, twist_number2d, h, k, w, tau)


plt.figure(1)
plt.clf()
plt.plot(tau_ring, omega_ring, 'r-', lw=2, label='ring parametric')
plt.plot(tau_square_periodic, omega_square_periodic, 'g-', lw=2, label='square periodic parametric')
plt.plot(tau2_ring, omega2_ring, 'ro', lw=2, label='ring')
plt.plot(tau2_square_periodic, omega2_square_periodic, 'go', lw=2, label='square periodic')
plt.xlim([0, 2])
#plt.ylim([4, 8])
plt.grid(True)
plt.legend(fontsize=9, loc='lower right')
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Frequency $\Omega$')
plt.tight_layout()



# ##############################################################################
# Store data
# ##############################################################################
print 'Store data'

# Figures
#mpl_helpers.grid_all()
#mpl_helpers.tight_layout_and_grid_all()
plt.show()