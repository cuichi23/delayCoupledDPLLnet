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

# Plotting
cmap = cm.coolwarm
dpi = 150


# ##############################################################################
# 1. Parametric frequency curves for different topologies for two coupled osci
# ##############################################################################
print '1. Parametric frequency curves for different topologies for two coupled osci'

n = 2
topo_global = synctools.Global(n)
topo_ring = synctools.Ring(n)
topo_chain = synctools.Chain(n)
twist_number = synctools.InPhase()
s_min =  0.0
s_max = 20.0
ds = 0.01

tau_global, omega_global, s_global = synctools.get_parametric_omega_curve2(topo_global, twist_number, h, k, w, s_min, s_max, ds)
tau_ring, omega_ring, s_ring = synctools.get_parametric_omega_curve2(topo_ring, twist_number, h, k, w, s_min, s_max, ds)
tau_chain, omega_chain, s_chain = synctools.get_parametric_omega_curve2(topo_chain, twist_number, h, k, w, s_min, s_max, ds)



plt.figure(1)
plt.clf()
plt.plot(tau_global, omega_global, 'b-', lw=2, label='global')
plt.plot(tau_ring, omega_ring, 'r-', lw=2, label='ring')
plt.plot(tau_chain, omega_chain, 'y-', lw=2, label='chain')
plt.xlim([0, 2])
plt.ylim([4, 8])
plt.grid(True)
plt.legend(fontsize=9)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Frequency $\Omega$')
plt.tight_layout()


# ##############################################################################
# 2. Parametric frequency curves for different topologies for 12 coupled osci
# ##############################################################################
print '2. Parametric frequency curves for different topologies for 12 coupled osci'

n = 12
nx = 4
ny = 3
topo_global = synctools.Global(n)
topo_ring = synctools.Ring(n)
topo_chain = synctools.Chain(n)
topo_square_periodic = synctools.SquarePeriodic(nx, ny)
topo_square_open = synctools.SquareOpen(nx, ny)
twist_number = synctools.InPhase()
s_min =  0.0
s_max = 20.0
ds = 0.01

tau_global, omega_global, s_global = synctools.get_parametric_omega_curve2(topo_global, twist_number, h, k, w, s_min, s_max, ds)
tau_ring, omega_ring, s_ring = synctools.get_parametric_omega_curve2(topo_ring, twist_number, h, k, w, s_min, s_max, ds)
tau_chain, omega_chain, s_chain = synctools.get_parametric_omega_curve2(topo_chain, twist_number, h, k, w, s_min, s_max, ds)
tau_square_periodic, omega_square_periodic, s_square_periodic = synctools.get_parametric_omega_curve2(topo_square_periodic, twist_number, h, k, w, s_min, s_max, ds)
tau_square_open, omega_square_open, s_square_open = synctools.get_parametric_omega_curve2(topo_square_open, twist_number, h, k, w, s_min, s_max, ds)


plt.figure(2)
plt.clf()
plt.plot(tau_global, omega_global, 'b-', lw=2, label='global')
plt.plot(tau_ring, omega_ring, 'r-', lw=2, label='ring')
plt.plot(tau_chain, omega_chain, 'y-', lw=2, label='chain')
plt.plot(tau_square_periodic, omega_square_periodic, 'g-', lw=2, label='square periodic')
plt.plot(tau_square_open, omega_square_open, 'm-', lw=2, label='square open')
plt.xlim([0, 2])
plt.ylim([4, 8])
plt.grid(True)
plt.legend(fontsize=9)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Frequency $\Omega$')
plt.tight_layout()


# ##############################################################################
# 3. Parametric m=1 twist frequency curves for different topologies for 12 coupled osci
# ##############################################################################
print '3. Parametric m=1 twist frequency curves for different topologies for 12 coupled osci'

n = 12
nx = 4
ny = 3
topo_ring = synctools.Ring(n)
topo_square_periodic = synctools.SquarePeriodic(nx, ny)
twist_number1d = synctools.Twist1D(1)
twist_number2d = synctools.Twist2D(1, 0)
s_min =  0.0
s_max = 20.0
ds = 0.01

tau_ring, omega_ring, s_ring = synctools.get_parametric_omega_curve2(topo_ring, twist_number1d, h, k, w, s_min, s_max, ds)
tau_square_periodic, omega_square_periodic, s_square_periodic = synctools.get_parametric_omega_curve2(topo_square_periodic, twist_number2d, h, k, w, s_min, s_max, ds)


plt.figure(3)
plt.clf()
plt.plot(tau_ring, omega_ring, 'r-', lw=2, label='ring')
plt.plot(tau_square_periodic, omega_square_periodic, 'g-', lw=2, label='square periodic')
plt.xlim([0, 2])
#plt.ylim([4, 8])
plt.grid(True)
plt.legend(fontsize=9)
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