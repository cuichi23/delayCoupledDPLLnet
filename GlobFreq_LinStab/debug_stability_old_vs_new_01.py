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
import scipy.optimize as optimize

import synctools
import synctools_old
reload(synctools)



# ##############################################################################
# Parameters
# ##############################################################################

# Parameters
w = (2 * np.pi) * 1.00
k = (2 * np.pi) * 0.25
h = synctools.Triangle(1.0 / (2 * np.pi))
#h = synctools.Sin(1.0 / (2 * np.pi))
dhdx = h.get_derivative()
wc = w
n = 7
topo1 = synctools.Ring(2)
twist1 = synctools.Twist1D(0)

# Plotting
cmap = cm.coolwarm
dpi = 150




# ##############################################################################
# 1. Reference stabilities 1
# ##############################################################################
print '1. Reference stabilities 1'


# New methods
tau1 = np.linspace(0, 2, 200)
tau1, omega1 = synctools.get_omega_curve(topo1, twist1, h, k, w, tau1)
tau1, l1 = synctools.get_stability_curve(w, k, h, wc, tau1, omega1, topo1, twist1)
c1 = np.real(l1)
vmax = np.max(np.abs(c1))
vmin = -vmax


# Old methods
l2 = []
for i_tau in range(len(tau1)):
    l_tmp = synctools.get_stability(n, 2, 1, w, k, h, 0, 0, 0, tau1[i_tau], omega1[i_tau], wc, 'ring')
    l2.append(l_tmp)
l2 = np.array(l2)
tau2 = tau1
c2 = np.real(l2)

# Old old methods
l3 = []
for i_tau in range(len(tau1)):
    l_tmp = synctools_old. get_stability(n, w, k, h, 0, tau1[i_tau], omega1[i_tau], wc)
    l3.append(l_tmp)
l3 = np.array(l3)
tau3 = tau1
c3 = np.real(l3)


# Manual calculation
tau4 = tau1
omega4 = omega1
d4 = np.zeros((n, n))
for i_row in range(n):
    i_minus = np.mod(i_row - 1, n)
    i_plus  = np.mod(i_row + 1, n)
    d4[i_row, i_minus] = 1.0 / n
    d4[i_row, i_plus] = 1.0 / n
em4, vm4 = np.linalg.eig(d4)
b = 1.0 / wc
l4 = []
for i_tau in range(len(tau4)):
    lambda_zeta = []
    for i_eigen in range(len(em4)):
        zeta = em4[i_eigen]
        a = k * dhdx(-tau4[i_tau] * omega4[i_tau])
        tau = tau4[i_tau]
        def func(l):
            alpha = np.real(zeta)
            beta  = np.imag(zeta)
            x = np.zeros(2)
            x[0] = l[0] + b * l[0]**2 - b * l[1]**2 + a - a * alpha * np.exp(-l[0] * tau) * np.cos(l[1] * tau) - a * beta * np.exp(-l[0] * tau) * np.sin(l[1] * tau)
            x[1] = l[1] + 2 * b * l[0] * l[1] + a * alpha * np.exp(-l[0] * tau) * np.sin(l[1] * tau) - a * beta * np.exp(-l[0] * tau) * np.cos(l[1] * tau)
            return x

        l_opt = optimize.root(func, np.array([0.01, 0.01]), tol=1e-14)
        l_tmp = l_opt.x[0] + 1j * l_opt.x[1]

        # Ignore solution for the eigenvector (a, a, a, ...)
        if np.max(np.abs(np.diff(vm4[:, i_eigen]))) >= 1e-9:
            lambda_zeta.append(l_tmp)

    lambda_zeta = np.array(lambda_zeta)
    l4.append(lambda_zeta[np.argmax(np.real(lambda_zeta))])
l4 = np.array(l4)
c4 = np.real(l4)


# Manual 2
tau5 = tau1
omega5 = omega1
d5 = d4
em5, vm5 = np.linalg.eig(d5)
b = 1.0 / wc
l5 = []
for i_tau in range(len(tau5)):
    lambda_zeta = []
    for i_eigen in range(len(em5)):
        zeta = em5[i_eigen]
        a = k * dhdx(-tau5[i_tau] * omega5[i_tau])
        tau = tau5[i_tau]
        def func(l):
            alpha = np.real(zeta)
            beta  = np.imag(zeta)
            x = np.zeros(2)
            x[0] = l[0] + b * l[0]**2 - b * l[1]**2 + a - a * alpha * np.exp(-l[0] * tau) * np.cos(l[1] * tau) - a * beta * np.exp(-l[0] * tau) * np.sin(l[1] * tau)
            x[1] = l[1] + 2 * b * l[0] * l[1] + a * alpha * np.exp(-l[0] * tau) * np.sin(l[1] * tau) - a * beta * np.exp(-l[0] * tau) * np.cos(l[1] * tau)
            x[0] = l[0] + b * l[0]**2 - b * l[1]**2 + a - a * alpha * np.exp(-l[0] * tau) * np.cos(l[1] * tau)
            x[1] = l[1] + 2 * b * l[0] * l[1] + alpha * np.exp(-l[0] * tau) * np.sin(l[1] * tau)
            x[0] = l[0] + b * l[0]**2 - b * l[1]**2 + a - a * zeta * np.exp(-l[0] * tau) * np.cos(l[1] * tau)
            x[1] = 2 * b * l[0] * l[1] + l[1] + a * zeta * np.exp(-l[0] * tau) * np.sin(l[1] * tau)
            return x

        l_opt = optimize.root(func, np.array([0.01, 0.01]), tol=1e-14)
        l_tmp = l_opt.x[0] + 1j * l_opt.x[1]

        # Ignore solution for the eigenvector (a, a, a, ...)
        if np.max(np.abs(np.diff(vm4[:, i_eigen]))) >= 1e-9:
            lambda_zeta.append(l_tmp)

    lambda_zeta = np.array(lambda_zeta)
    l5.append(lambda_zeta[np.argmax(np.real(lambda_zeta))])
l5 = np.array(l5)
c5 = np.real(l5)


# Figure
plt.figure(1)
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
#plt.scatter(tau1, c1, c=c1, cmap=cmap, edgecolors='face', vmin=vmin, vmax=vmax, label='new')
#plt.plot(tau1, c2, 'yo', mec='y', label='synctools old')
#plt.plot(tau3, c3, 'ko', mec='k', label='synctools initial')
plt.plot(tau4, c4, 'mo', mec='m', label='manual')
plt.plot(tau5, c5, 'co', mec='c', label='njp')
plt.axhline(0.0, color='k')
plt.xlim([0, 2])
#plt.ylim([-0.5, 1.5])
plt.grid(True)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Stability $\mathrm{Re}(\lambda)$')
plt.legend(fontsize=9)

plt.tight_layout()



# ##############################################################################
# 2. Investigation at single point
# ##############################################################################
print '2. Investigation at single point'

tau_single = tau1[100]
omega_single = omega1[100]

l1_single = synctools.get_stability2(w, k, h, wc, tau_single, omega_single, topo1, twist1)
l2_single = synctools.get_stability(n, 2, 1, w, k, h, 0, 0, 0, tau_single, omega_single, wc, 'ring')

d1= topo1.get_couling_derivate_matrix(h, twist1, omega_single * tau_single)
d2, alpha_plus, alpha_minus = synctools.calcTopoMatrix(n, 2, 1, w, k, h, 0, 0, 0, tau_single, omega_single, wc, 'ring')
em1, vm1 = np.linalg.eig(d1)
em2, vm2 = np.linalg.eig(d2)


# Manual calculation
tau4 = tau1
omega4 = omega1
d4 = np.zeros((n, n))
for i_row in range(n):
    i_minus = np.mod(i_row - 1, n)
    i_plus  = np.mod(i_row + 1, n)
    d4[i_row, i_minus] = 1.0 / n
    d4[i_row, i_plus] = 1.0 / n
em4, vm4 = np.linalg.eig(d4)
b = 1.0 / wc
l4 = []
lambda_zeta = []
for i_eigen in range(len(em4)):
    zeta = em4[i_eigen]
    a = k * dhdx(-tau_single * omega_single)
    tau = tau_single
    def func(l):
        alpha = np.real(zeta)
        beta  = np.imag(zeta)
        x = np.zeros(2)
        x[0] = l[0] + b * l[0]**2 - b * l[1]**2 + a - a * alpha * np.exp(-l[0] * tau) * np.cos(l[1] * tau) - a * beta * np.exp(-l[0] * tau) * np.sin(l[1] * tau)
        x[1] = l[1] + 2 * b * l[0] * l[1] + a * alpha * np.exp(-l[0] * tau) * np.sin(l[1] * tau) - a * beta * np.exp(-l[0] * tau) * np.cos(l[1] * tau)
        return x

    l_opt = optimize.root(func, np.array([0.01, 0.01]), tol=1e-14)
    l_tmp = l_opt.x[0] + 1j * l_opt.x[1]

    # Ignore solution for the eigenvector (a, a, a, ...)
    if np.max(np.abs(np.diff(vm4[:, i_eigen]))) >= 1e-9:
        lambda_zeta.append(l_tmp)

lambda_zeta = np.array(lambda_zeta)
l4_single = lambda_zeta[np.argmax(np.real(lambda_zeta))]



# ##############################################################################
# Store data
# ##############################################################################
print 'Store data'

# Figures
#mpl_helpers.grid_all()
#mpl_helpers.tight_layout_and_grid_all()
plt.show()