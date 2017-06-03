# ##############################################################################
# Object-oriented parameter sweep
# Version 1: Initial version
# Version 2: Module import
# Author: Daniel Platz
# ##############################################################################
import os

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize

import synctools



# ##############################################################################
# Parameters
# ##############################################################################

# Export
filename_fig1 = 'freq_stab_tau_sweep.png'

# Plotting
cmap = cm.coolwarm
dpi = 150




# ##############################################################################
# 1. Demo PLL, SweepFactory and FlatStateList objects
# ##############################################################################
print '1. Demo PLL, SweepFactory and FlatStateList objects'

# System parameters
n = 3
w = 2 * np.pi
k = 1.57
wc = 1 * w
m = 1
tau = np.linspace(0, 2, 250)
h = synctools.Triangle(1.0 / (2 * np.pi))
tsim = 0.1234


# Create a pll system and determine the possible m-twist states
pll = synctools.PllSystem(n, w, k, tau[0], h, wc)
s = pll.get_twist_state(m)

# Perform a delay sweep
sf = synctools.SweepFactory(n, w, k, tau, h, wc, m, tsim=tsim)
fsl = sf.sweep()

# Extract variables from the sweep
para_mat = fsl.get_parameter_matrix()
tau2 = fsl.get_tau()
omega2 = fsl.get_omega()
l2 = fsl.get_l()
c_l2 = np.real(l2)
vmin = -np.max(np.abs(c_l2))
vmax = -vmin



# Figure 1: Plot global frequency and linear perturbation decay rate from delay sweep
plt.figure(1)
plt.clf()
gs = gridspec.GridSpec(2, 1)
xlim = [0, 2]

plt.subplot(gs[0])
plt.title(r'System parameters: n = %i, k = %.f, $\omega$ = %.3f, $\omega_c$ = %.3f' % (n, k, w, wc))
plt.scatter(tau2, omega2, c=c_l2, cmap=cmap, edgecolors='face', vmin=vmin, vmax=vmax)
plt.colorbar(orientation='horizontal')
plt.legend(fontsize=9)
plt.xlim(xlim)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Global frequency $\Omega$')
plt.grid(True)


plt.subplot(gs[1])
plt.plot(tau2, np.real(l2), 'b.', lw=2)
plt.axhline(0.0, color='k')
plt.xlim(xlim)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Decay constant Re($\lambda$)')
plt.grid(True)
plt.tight_layout()



# ##############################################################################
# 2. Clean up
# ##############################################################################
print '2. Clean up'

# Figures
plt.figure(1)
plt.savefig(filename_fig1, dpi=dpi)
plt.show()