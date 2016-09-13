# ##############################################################################
# Object-oriented parameter sweep
# Version 1: Initial version
# Author: Daniel Platz
# ##############################################################################
import os

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize



# ##############################################################################
# Parameters
# ##############################################################################

# Export
filename_fig1 = 'freq_stab_tau_sweep.png'

# Plotting
cmap = cm.coolwarm
dpi = 150


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

# ##############################################################################

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


# ##############################################################################


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

# ##############################################################################


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


def get_omega_implicit(n, w, k, tau, h, m):
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

class TestClass(object):
    def __init__(self):
        self.a = 1.0
        self.b = np.linspace(0, 1, 11)


# ##############################################################################

class PllSystem(object):
    def __init__(self, n, w, k, tau, h, wc):
        self.n = n
        self.w = w
        self.k = k
        self.tau = tau
        self.h = h
        self.wc = wc

    def get_twist_state(self, m):
        o = get_omega_implicit(self.n, self.w, self.k, self.tau, self.h, m)
        if o != None:
            s = []
            for el in o:
                l = get_stability(self.n, self.w, self.k, self.h, m, self.tau, el, self.wc)
                s.append(TwistState(self, m, el, l))
            return s
        else:
            return None



class TwistState(object):
    def __init__(self, system, m, omega, l):
        self.m = m
        self.omega = omega
        self.l = l
        self.system = system


class FlatStateList(object):
    def __init__(self):
        self.states = []
        self.n = 0

    def add_states(self, s):
        if type(s) is TwistState:
            self.states.append(s)
            self.n = len(self.states)
        elif type(s) is list:
            for el in s:
                self.states.append(el)
            self.n = len(self.states)

    def get_n(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].system.n
            return x
        else:
            return None


    def get_w(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].system.w
            return x
        else:
            return None

    def get_k(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].system.k
            return x
        else:
            return None

    def get_tau(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].system.tau
            return x
        else:
            return None

    def get_h(self):
        if self.n > 0:
            x = []
            for i in range(self.n):
                x.append(self.states[i].system.h)
            return x
        else:
            return None

    def get_m(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].m
            return x
        else:
            return None

    def get_omega(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].omega
            return x
        else:
            return None

    def get_l(self):
        if self.n > 0:
            x = np.zeros(self.n, dtype=np.complex)
            for i in range(self.n):
                x[i] = self.states[i].l
            return x
        else:
            return None

    def get_parameter_matrix(self):
        if self.n > 0:
            x = np.zeros((self.n, 8))
            x[:, 0] = self.get_n()
            x[:, 1] = self.get_w()
            x[:, 2] = self.get_k()
            x[:, 3] = self.get_tau()
            x[:, 4] = self.get_m()
            x[:, 5] = self.get_omega()
            x[:, 6] = np.real(self.get_l())
            x[:, 7] = np.imag(self.get_l())
            return x
        else:
            return None



class SweepFactory(object):
    def __init__(self, n, w, k, tau, h, wc, m):
        self.n = n
        self.w = w
        self.k = k
        self.tau = tau
        self.h = h
        self.wc = wc
        self.m = m

    def _identify_swept_variable(self):
        if type(self.n) is np.ndarray:
            return 'n'
        elif type(self.w) is np.ndarray:
            return 'w'
        elif type(self.k) is np.ndarray:
            return 'k'
        elif type(self.w) is np.ndarray:
            return 'w'
        elif type(self.tau) is np.ndarray:
            return 'tau'
        elif type(self.wc) is np.ndarray:
            return 'wc'
        else:
            return None

    def __getitem__(self, key):
        return self.__dict__[key]

    def sweep(self):
        fsl = FlatStateList()
        key_sweep = self._identify_swept_variable()
        par_sweep = self[key_sweep]
        n_sweep = len(par_sweep)
        key_sys = ['n', 'w', 'k', 'tau', 'h', 'wc']
        for i in range(n_sweep):
            args = []
            for key in key_sys:
                if key == key_sweep:
                    args.append(self[key][i])
                else:
                    args.append(self[key])
            pll = PllSystem(*args)
            s = pll.get_twist_state(self.m)
            fsl.add_states(s)
        return fsl








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
h = Triangle(1.0 / (2 * np.pi))


# Create a pll system and determine the possible m-twist states
pll = PllSystem(n, w, k, tau[0], h, wc)
s = pll.get_twist_state(m)

# Perform a delay sweep
sf = SweepFactory(n, w, k, tau, h, wc, m)
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
plt.colorbar()
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