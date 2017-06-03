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
wc = w

n = 12
nx = 4
ny = 3
topo_global = synctools.Global(n)
topo_ring = synctools.Ring(n)
topo_square_periodic = synctools.SquarePeriodic(nx, ny)
topo_square_open = synctools.SquareOpen(nx, ny)
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
tau = np.linspace(0, 2, 50)
tau2_ring, omega2_ring = synctools.get_omega_curve(topo_ring, twist_number1d, h, k, w, tau)
tau2_square_periodic, omega2_square_periodic = synctools.get_omega_curve(topo_square_periodic, twist_number2d, h, k, w, tau)


plt.figure(1)
plt.clf()
gs1 = plt.GridSpec(2, 1)

plt.subplot(gs1[0])
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
# 2. Build coupling derivative matrices
# ##############################################################################
print '2. Build coupling derivative matrices'

in_phase = synctools.InPhase()
twist = synctools.Twist1D(1)
d_global          = topo_global.get_couling_derivate_matrix(h, in_phase, 0.21)
d_ring            = topo_ring.get_couling_derivate_matrix(h, twist_number1d, 0.21)
d_square_periodic = topo_square_periodic.get_couling_derivate_matrix(h, twist_number2d, 0.21)
d_square_open     = topo_square_open.get_couling_derivate_matrix(h, in_phase, 0.21)


# ##############################################################################
# 3. Compute stabilities
# ##############################################################################
print '3. Compute stabilities'


tau3_ring, l_ring                       = synctools.get_stability_curve(w, k, h, wc, tau2_ring, omega2_ring, topo_ring, twist_number1d)
tau3_square_periodic, l_square_periodic = synctools.get_stability_curve(w, k, h, wc, tau2_square_periodic, omega2_square_periodic, topo_square_periodic, twist_number2d)

plt.figure(1)

plt.subplot(gs1[1])
plt.plot(tau3_ring, np.real(l_ring), 'ro', lw=2, label='ring')
plt.plot(tau3_square_periodic, np.real(l_square_periodic), 'go', lw=2, label='square periodic')
plt.axhline(0.0, color='k')
plt.xlim([0, 2])
#plt.ylim([4, 8])
plt.grid(True)
plt.legend(fontsize=9, loc='lower right')
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Frequency $\Omega$')
plt.tight_layout()


# ##############################################################################
# 4. Reference stabilities 1
# ##############################################################################
print '4. Reference stabilities 1'

w = 2 * np.pi * 1.00
k = 2 * np.pi * 0.25
wc = 1.0 * w
topo1 = synctools.Ring(2)
twist1 = synctools.Twist1D(0)
tau1 = np.linspace(0, 2, 200)
tau1, omega1 = synctools.get_omega_curve(topo1, twist1, h, k, w, tau1)
tau1, l1 = synctools.get_stability_curve(w, k, h, wc, tau1, omega1, topo1, twist1)
c1 = np.real(l1)
vmax = np.max(np.abs(c1))
vmin = -vmax

plt.figure(2)
plt.clf()
gs2 = plt.GridSpec(2, 1)

plt.subplot(gs2[0])
plt.scatter(tau1, omega1, c=c1, cmap=cmap, edgecolors='face', vmin=vmin, vmax=vmax)
plt.colorbar(orientation='horizontal')
plt.xlim([0, 2])
plt.ylim([4, 8])
plt.grid(True)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Frequency $\Omega$')

plt.subplot(gs2[1])
plt.scatter(tau1, c1, c=c1, cmap=cmap, edgecolors='face', vmin=vmin, vmax=vmax)
plt.axhline(0.0, color='k')
plt.xlim([0, 2])
#plt.ylim([4, 8])
plt.grid(True)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Stability $\mathrm{Re}(\lambda)$')

plt.tight_layout()


# ##############################################################################
# Store data
# ##############################################################################
print 'Store data'

# Figures
#mpl_helpers.grid_all()
#mpl_helpers.tight_layout_and_grid_all()
plt.show()