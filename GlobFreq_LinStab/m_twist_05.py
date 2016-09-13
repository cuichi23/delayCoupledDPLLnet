# ##############################################################################
# Version 3: Numerical solution for eigenvalues of m-twist coupling matrix
# Version 4: Function encapsulation
# Version 5: Stability for parametric and implicit freq curve calculated with same function
# Author: Daniel Platz
# ##############################################################################
import os

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal

#import my_paths2
#import hdf5
#import mpl_helpers

#clear_all()


# ##############################################################################
# Parameters
# ##############################################################################

# System parameters
#w = 6.28
#k = 1.57
#n = 3
#wc = 100 * w
#m = 1


w = 2 * np.pi
k = 1.57
n = 3
wc = 1 * w
m = 0

s_min = 0.0
s_max = 20.0
ds = 0.01


# Export parameters
filename_fig1 = 'triangle_function.png'
filename_fig2 = 'm_twist_parametric_freq_curve.png'
filename_fig3 = 'm_twist_root_freq_curve.png'
filename_fig4 = 'm_twist_linear_stability.png'
filename_fig5 = 'm_twist_linear_stability_binary.png'
filename_fig6 = 'm_twist_linear_stability_continuous.png'
filename_export = 'results_root.hdf5'


# Plotting
cmap = cm.coolwarm
dpi = 150
xlim = [0, 1.7]
ylim = [4.5, 8.0]

# ##############################################################################
# Definitions
# ##############################################################################
def tight_layout_and_grid_all(isVerbose=True):
    ''' Activates tight_layout for all open axes '''
    fignums = plt.get_fignums()
    for num in fignums:
        if isVerbose:
            print 'Formatting figure %i\r' % num,
        fig = plt.figure(num)
        if len(fig.get_axes()) > 0:
            for ax in fig.get_axes():
                ax.grid(True)
            plt.draw()
            plt.tight_layout()
            plt.draw()


def clear_all():
    ''' Clears all figures currently displayed '''
    fignums = plt.get_fignums()
    for num in fignums:
        plt.figure(num)
        plt.clf()


class Square(object):
    def __init__(self, freq, amp):
        self.freq = freq
        self.amp = amp

    def __call__(self, t):
        return self.amp * signal.square(2 * np.pi * self.freq * t, duty=0.5)


class Triangle(object):
    def __init__(self, freq):
        self.freq = freq

    def __call__(self, t):
        return signal.sawtooth(2 * np.pi * self.freq * t, width=0.5)

    def get_derivative(self):
        amp = 2.0 * (2 * self.freq)
        return Square(self.freq, amp)

    def max(self):
        return 1.0

    def min(self):
        return -1.0


class SingleOmega(object):
    def __init__(self, tau, omega):
        self.tau = tau
        self.omega = omega


class ParametricOmega(object):
    def __init__(self, tau, omega, s):
        self.tau = tau
        self.omega = omega
        self.s = s


class MultiOmega(object):
    def __init__(self, tau, omega):
        self.tau = tau
        self.omega = omega

    def get_single_omega(self):
        tau_tmp = []
        omega_tmp = []
        for it in range(len(self.tau)):
            for el in self.omega[it]:
                tau_tmp.append(self.tau[it])
                omega_tmp.append(el)
        tau_tmp = np.array(tau_tmp)
        omega_tmp = np.array(omega_tmp)
        return SingleOmega(tau_tmp, omega_tmp)


def get_sign_changes(x):
    ''' Returns an integer array with the indices of the element in x before a
        sign change.
    '''
    i = np.where(np.diff(np.sign(x)))
    if len(i) > 0:
        return i[0]
    else:
        return []


def get_parametric_omega_curve(n, w, k, h, m, s_min, s_max, ds):
    phi_m = (2 * np.pi * m) / n
    n_s = np.int(np.rint((s_max - s_min) / ds))
    s = np.linspace(s_min, s_max, n_s)
    omega = w + (k / 2.0) * (h(-s + phi_m) + h(-s - phi_m))
    tau  = s / omega

    return ParametricOmega(tau, omega, s)


def _get_implicit_omega(n, w, k, h, m, tau):
    phi_m = (2 * np.pi * m) / n
    func = lambda s: -s + 0.5 * k * tau * (h(-s + phi_m) + h(-s - phi_m)) + w * tau

    h_min = 2 * h.min()
    h_max = 2 * h.max()
    s_min = h_min + w * tau
    s_max = h_max + w * tau
    s = np.linspace(s_min, s_max, 10000)

    i_root = get_sign_changes(func(s))
    if len(i_root) > 0:
        omega = []
        for ir in range(len(i_root)):
            s_tmp = optimize.brentq(func, s[i_root[ir]], s[i_root[ir] + 1])
            omega.append(w + 0.5 * k * (h(-s_tmp + phi_m) + h(-s_tmp - phi_m)))
        return omega
    else:
        return []


def get_implicit_omega(n, w, k, h, m, tau):
    if type(tau) == np.ndarray:
        omega = []
        for it in range(len(tau)):
            omega.append(_get_implicit_omega(n, w, k, h, m, tau[it]))
        return MultiOmega(tau, omega)
    elif type(tau) == float or type(tau) == int or type(tau) == np.float or type(tau) == np.float64:
        omega = _get_implicit_omega(n, w, k, h, m, tau)
        return omega
    else:
        print 'Wrong datatype for tau'
        return None


def get_stability(n, w, k, h, m, tau, omega, wc):
    # Dependent parameters
    b = 1.0 / wc
    dhdt = h.get_derivative()
    phi_m = (2 * np.pi * m) / n

    # Compute stability for root-based frequencies
    alpha_plus = k * dhdt(-omega * tau - phi_m)
    alpha_minus = k * dhdt(-omega * tau + phi_m)

    e_mat = np.zeros((n, n))
    e_mat[0, -1] = alpha_plus
    e_mat[0, 1] = alpha_minus
    for ik in range(1, n - 1):
        e_mat[ik, ik - 1] = alpha_plus
        e_mat[ik, ik + 1] = alpha_minus
    e_mat[-1, 0] = alpha_minus
    e_mat[-1, -2] = alpha_plus
    em, vm = np.linalg.eig(e_mat)

    lambda_nu = []
    for inu in range(len(em)):
        nu = em[inu]
        #print 'tau = %.3f: %.6f + 1j * %.6f' % (tau, np.real(nu), np.imag(nu))
        def func(l):
            mu = np.real(nu)
            gamma = np.imag(nu)
            x = np.zeros(2)
            x[0] = b * l[0]**2 - b * l[1]**2 + l[0] + 0.5 * (alpha_plus + alpha_minus) - 0.5 * mu * np.exp(-l[0] * tau) * np.cos(l[1] * tau) - 0.5 * gamma * np.exp(-l[0] * tau) * np.sin(l[1] * tau)
            x[1] = 2 * b * l[0] * l[1] + l[1] + 0.5 * mu * np.exp(-l[0] * tau) * np.sin(l[1] * tau) - 0.5 * gamma * np.exp(-l[0] * tau) * np.cos(l[1] * tau)
            return x

        l_opt = optimize.root(func, np.array([1.0, 1.0]), tol=1e-15)
        l_tmp = l_opt.x[0] + 1j * l_opt.x[1]
        if np.max(np.abs(np.diff(vm[:, inu]))) >= 1e-9:
            lambda_nu.append(l_tmp)

    lambda_nu = np.array(lambda_nu)
    return lambda_nu[np.argmax(np.real(lambda_nu))]







# ##############################################################################
# 1. Define triangle wave
# ##############################################################################
print '1. Define triangle wave'
clear_all()

h = Triangle(1.0 / (2 * np.pi))
dhdt = h.get_derivative()
#h = Triangle(0.2)
t = np.linspace(-4 * np.pi, 4 * np.pi, 2000)

plt.figure(1)
plt.title('The (digital) triangle function')
plt.plot(t / np.pi, h(t), 'b-')
plt.plot(t / np.pi, dhdt(t), 'r-')
#plt.ylim([-1.2, 1.2])
plt.xlabel('Time in units of pi')
plt.ylabel('Triangle function')


# ##############################################################################
# 2. Parametric global frequency curve
# ##############################################################################
print '2. Parametric global frequency curve'


phi_m = (2 * np.pi * m) / n
s = np.linspace(0, 20, 2000)
omega_para = w + (k / 2.0) * (h(-s + phi_m) + h(-s - phi_m))
tau_para  = s / omega_para

plt.figure(2)
plt.title('Global synchronization frequency as a paranetric curve')
plt.plot(tau_para, omega_para, 'b.')
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Global frequency $\Omega$')
plt.xlim(xlim)




# ##############################################################################
# 3. Self-consistent calculation of global frequency
# ##############################################################################
print '3. Self-consistent calculation of global frequency'

# Compute global frequency from implicit equation
tau_root = np.linspace(0, 2, 50)
o_root = get_implicit_omega(n, w, k, h, m, tau_root)
o_root2 = o_root.get_single_omega()

# Parametric omega curve
o_para = get_parametric_omega_curve(n, w, k, h, m, s_min, s_max, ds)


# Figure 3
plt.figure(3)
plt.plot(o_para.tau, o_para.omega, 'b-', label='parametric', lw=2)
for it in range(len(o_root.tau)):
    plt.axvline(o_root.tau[it], color='k', alpha=0.3)
    if len(o_root.omega[it]) > 0:
        for ir in range(len(o_root.omega[it])):
            l = plt.plot(o_root.tau[it], o_root.omega[it][ir], 'ro')[0]
l.set_label('root')
plt.plot(o_root2.tau, o_root2.omega, 'k.', label='root (flat)')


plt.legend(fontsize=9)
plt.xlim(xlim)

plt.title('System parameters: n = %i, m=%i, k = %.3f, w = %.3f' % (n, m, k, w))
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Global frequency $\Omega$')





# ##############################################################################
# 4. Determine linear stability of states
# ##############################################################################
print '4. Determine linear stability'

# Compute stability for root-based frequencies
lambda_root = []
for it in range(len(o_root2.tau)):
    lambda_root.append(get_stability(n, w, k, h, m, o_root2.tau[it], o_root2.omega[it], wc))
lambda_root = np.array(lambda_root)




# Compute stability for parametric-based frequencies
lambda_para = np.zeros(len(o_para.tau), dtype=np.complex)
for it in range(len(o_para.tau)):
    lambda_para[it] = get_stability(n, w, k, h, m, o_para.tau[it], o_para.omega[it], wc)


# Figure 4
plt.figure(4)
gs = gridspec.GridSpec(2, 1)

plt.subplot(gs[0])
plt.title('System parameters: n = %i, k = %.f, w = %.3f, wc = %.3f' % (n, k, w, wc))
plt.plot(o_para.tau, o_para.omega, 'b-', label='parametric', lw=2)
plt.plot(o_root2.tau, o_root2.omega, 'ro', label='root')
plt.legend(fontsize=9)
plt.xlim(xlim)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Global frequency $\Omega$')


plt.subplot(gs[1], sharex=plt.gca())
plt.plot(o_para.tau, np.real(lambda_para), 'b.', label='parametric', lw=2)
plt.plot(o_root2.tau, np.real(lambda_root), 'ro', label='root')
plt.axhline(0.0, color='k')
plt.xlim(xlim)
plt.ylim([-5, 2])
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Decay constant Re($\lambda$)')



# ##############################################################################
# 5. Scatter plot with binary color coding
# ##############################################################################
print '5. Scatter plot with binary color coding'

c_para = np.zeros(len(o_para.tau))
for it in range(len(o_para.tau)):
    if np.real(lambda_para[it]) <= 0:
        c_para[it] = 0.0
    else:
        c_para[it] = 1.0


plt.figure(5)
gs = gridspec.GridSpec(2, 1)

plt.subplot(gs[0])
plt.title('System parameters: n = %i, k = %.f, w = %.3f, wc = %.3f' % (n, k, w, wc))
plt.scatter(o_para.tau, o_para.omega, c=c_para, cmap=cmap, edgecolors='face')
plt.legend(fontsize=9)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Global frequency $\Omega$')


plt.subplot(gs[1])
plt.plot(o_para.tau, np.real(lambda_para), 'b.', label='parametric', lw=2)
plt.axhline(0.0, color='k')
plt.xlim(xlim)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Decay constant Re($\lambda$)')




# ##############################################################################
# 6. Scatter plot with continuous color coding
# ##############################################################################
print '6. Scatter plot with continuous color coding'

c_para = np.real(lambda_para)
vmin = -np.max(np.abs(c_para))
vmax = -vmin

plt.figure(6)
gs = gridspec.GridSpec(2, 1)

plt.subplot(gs[0])
plt.title('System parameters: n = %i, k = %.f, w = %.3f, wc = %.3f' % (n, k, w, wc))
plt.scatter(o_para.tau, o_para.omega, c=c_para, cmap=cmap, edgecolors='face', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.legend(fontsize=9)
plt.xlim(xlim)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Global frequency $\Omega$')


plt.subplot(gs[1])
plt.plot(o_para.tau, np.real(lambda_para), 'b.', label='parametric', lw=2)
plt.axhline(0.0, color='k')
plt.xlim(xlim)
plt.xlabel(r'Delay $\tau$')
plt.ylabel(r'Decay constant Re($\lambda$)')



# ##############################################################################
# Store data
# ##############################################################################
print 'Store data'


# Figures
#mpl_helpers.grid_all()
tight_layout_and_grid_all()
plt.figure(1)
plt.savefig(filename_fig1, dpi=dpi)
plt.figure(2)
plt.savefig(filename_fig2, dpi=dpi)
plt.figure(3)
plt.savefig(filename_fig3, dpi=dpi)
plt.figure(4)
plt.savefig(filename_fig4, dpi=dpi)
plt.figure(5)
plt.savefig(filename_fig5, dpi=dpi)
plt.figure(6)
plt.savefig(filename_fig6, dpi=dpi)
plt.show()