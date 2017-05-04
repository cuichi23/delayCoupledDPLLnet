# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:36:06 2017

@author: platz
"""
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal


# #############################################################################
# Helper functions
# #############################################################################

def get_sign_changes(x):
    ''' Returns an integer array with the indices of the element in x before a
        sign change.
    '''
    i = np.where(np.diff(np.sign(x)))
    if len(i) > 0:
        return i[0]
    else:
        return []




def wrap2pi(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi





# #############################################################################
# Geometry classes describing the virtual arrangement of the oscillators
# #############################################################################

class Arrangement(object):
    def _check_coordinate_type(self, yx):
        raise NotImplementedError

    def site_exists(self, yx):
        raise NotImplementedError

    def index2coordinate(self, i):
        raise NotImplementedError

    def coordinate2index(self, yx):
        raise NotImplementedError

    def get_n(self):
        raise NotImplementedError

    def get_dimensionality(self):
        raise NotImplementedError

    def get_shape(self):
        raise NotImplementedError

    def wrap_coordinates(self, yx):
        raise NotImplementedError

    def reshape_index_array(self, ii):
        raise NotImplementedError

    def flatten_spatial_matrix(self, mat):
        raise NotImplementedError




class Linear(Arrangement):
    def __init__(self, n):
        self.n = n

    def get_dimensionality(self):
        return 1

    def get_n(self):
        return self.n

    def get_shape(self):
        return (self.n)

    def reshape_index_array(self, ii):
        return ii

    def flatten_spatial_matrix(self, mat):
        return mat

    def index2coordinate(self, i):
        return i

    def coordinate2index(self, yx):
        return yx



class Ring(Linear):
    def site_exists(self, yx):
        return True

    def wrap_coordinates(self, yx):
        return int(np.mod(yx, self.n))



class Chain(Linear):
    def site_exists(self, yx):
        if yx >= 0 and yx < self.n:
            return True
        else:
            return False

    def wrap_coordinates(self, yx):
        if yx >= 0 and yx < self.n:
            return yx
        else:
            return None



class Cubic2D(Arrangement):
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    def get_dimensionality(self):
        return 2

    def get_n(self):
        return self.nx * self.ny

    def get_shape(self):
        return (self.ny, self.nx)

    def reshape_index_array(self, ii):
        shape = self.get_shape()
        return np.reshape(ii, shape)

    def flatten_spatial_matrix(self, mat):
        return mat.flatten()

    def index2coordinate(self, i):
        x = int(np.mod(i, self.nx))
        y = int(np.floor(i / float(self.nx)))
        return np.array([y, x], dtype=np.int32)

    def coordinate2index(self, yx):
        return self.nx * yx[0] + yx[0]



class PeriodicCubic2D(Cubic2D):
    def site_exists(self, yx):
        return True

    def wrap_coordinates(self, yx):
        y_wrap = int(np.mod(yx[0], self.ny))
        x_wrap = int(np.mod(yx[1], self.nx))
        return np.array([y_wrap, x_wrap], dtype=np.int)



class OpenCubic2D(Cubic2D):
    def site_exists(self, yx):
        if yx[0] >= 0 and yx[0] < self.ny and yx[1] >= 0 and yx[1] < self.nx:
            return True
        else:
            return False

    def wrap_coordinates(self, yx):
        if yx[0] >= 0 and yx[0] < self.ny and yx[1] >= 0 and yx[1] < self.nx:
            return yx
        else:
            return None






# #############################################################################
# Coupling functions
# #############################################################################

class CouplingFunction(object):
    def __call__(self, x):
        raise NotImplementedError

    def get_derivative(self):
        raise NotImplementedError

    def max(self):
        raise NotImplementedError

    def min(self):
        raise NotImplementedError


class Triangle(CouplingFunction):
    ''' Periodic triangle signal vertically centered around 0'''
    def __init__(self, freq=1.0 / (2 * np.pi)):
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


class Square(CouplingFunction):
    '''Periodic square wave vertically centered around 0'''
    def __init__(self, freq, amp):
        self.freq = freq
        self.amp = amp

    def __call__(self, t):
        return self.amp * signal.square(2 * np.pi * self.freq * t, duty=0.5)

    def max(self):
        return self.amp

    def min(self):
        return -self.amp




# #############################################################################
# Graph class describing the coupling
# #############################################################################

class Graph(object):
    def __init__(self, arrangement, function, strength, delay, hasNormalizedCoupling):
        self.arr = arrangement
        self.func = function
        self.k = strength
        self.tau = delay
        self.hasNormalizedCoupling = hasNormalizedCoupling

    def get_single_site_coupling(self, i):
        # Determine coupled sites in coordinate space
        yx = self.arr.index2coordinate(i)
        yx_c = yx + self.d

        # Check boundary conditions and transform to index space
        i_c = []
        for el in yx_c:
            if self.arr.site_exists(el):
                tmp = self.arr.wrap_coordinates(el)
                tmp = self.arr.coordinate2index(tmp)
                i_c.append(tmp)
        i_c = np.array(i_c)

        # Construct coupling vector in index space
        v = np.zeros(self.arr.get_n())
        for el in i_c:
            v[el] = 1.0

        # Normalize coupling by number of neighbors
        if self.hasNormalizedCoupling:
            v = v / np.sum(v)

        return v

    def get_single_site_coupling_matrix(self, yx):
        i = self.arr.coordinate2index(yx)
        v = self.get_single_site_coupling(i)
        m = self.arr.reshape_index_array(v)
        return m

    def get_coupling_matrix(self):
        n = self.arr.get_n()
        m = np.zeros((n, n))
        for ii in range(n):
            m[ii, :] = self.get_single_site_coupling(ii)
        return m


class NearestNeighbor(Graph):
    def __init__(self, arrangement, function, strength, delay, hasNormalizedCoupling):
        if isinstance(arrangement, Linear):
            super(NearestNeighbor, self).__init__(arrangement, function, strength, delay, hasNormalizedCoupling)
            self.d = np.array([-1, 1])
        else:
            raise Exception('Incompatible spatial lattice class')


class CubicNearestNeighbor(Graph):
    def __init__(self, arrangement, function, strength, delay, hasNormalizedCoupling):
        if isinstance(arrangement, Cubic2D):
            super(CubicNearestNeighbor, self).__init__(arrangement, function, strength, delay, hasNormalizedCoupling)
            d = []
            d.append([-1,  0])
            d.append([ 0, -1])
            d.append([ 0,  1])
            d.append([ 1,  0])
            self.d = np.array(d)
        else:
            raise Exception('Incompatible spatial lattice class')


class CubicHexagonal(Graph):
    def __init__(self, arrangement, function, strength, delay, hasNormalizedCoupling):
        if isinstance(arrangement, Cubic2D):
            super(CubicHexagonal, self).__init__(arrangement, function, strength, delay, hasNormalizedCoupling)
            d = []
            d.append([-1, -1])
            d.append([-1,  0])
            d.append([ 0, -1])
            d.append([ 0,  1])
            d.append([ 1,  0])
            d.append([ 1,  1])
            self.d = np.array(d)
        else:
            raise Exception('Incompatible spatial lattice class')


class CubicOctagonal(Graph):
    def __init__(self, arrangement, function, strength, delay, hasNormalizedCoupling):
        if isinstance(arrangement, Cubic2D):
            super(CubicOctagonal, self).__init__(arrangement, function, strength, delay, hasNormalizedCoupling)
            d = []
            d.append([-1, -1])
            d.append([-1,  0])
            d.append([-1,  1])
            d.append([ 0, -1])
            d.append([ 0,  1])
            d.append([ 1, -1])
            d.append([ 1,  0])
            d.append([ 1,  1])
            self.d = np.array(d)
        else:
            raise Exception('Incompatible spatial lattice class')



# #############################################################################
# Pll class
# #############################################################################

class Pll(object):
    def __init__(self, w, wc):
        self.w = w
        self.wc =wc
        self.b = 1.0 / wc


# #############################################################################
# Pll system class
# #############################################################################

class PllSystem(object):
    def __init__(self, pll, graph):
        self.pll = pll
        self.g = graph




# #############################################################################
# State definition classes
# #############################################################################

class SyncStateDefinition(object):
    def __init__(self, system):
        self.sys = system

    def get_phi(self):
        raise NotImplementedError

    def get_dphi_matrix(self):
        '''The static phase difference matrix

           dphi[i, j] = phi[j] - phi[i]
        '''
        n = self.sys.g.arr.get_n()
        phi = self.get_phi()
        dphi = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dphi[i, j] = phi[j] - phi[i]
        return dphi

    def get_states(self):
        sf = SyncStateFactory(self)
        return sf.get_states()


class CheckerboardDefinition(SyncStateDefinition):
    def __init__(self, system):
        if isinstance(system.g.arr, Chain):
            super(CheckerboardDefinition, self).__init__(system)
        else:
            raise Exception('State definition not compatible with system.')

    def get_phi(self):
        '''The static phases of all oscillators

           The phase of oscillator 0 is set to 0.
        '''
        n = self.sys.g.arr.get_n()
        phi = np.zeros(n)
        for i in range(n):
           phi[i] = ((-1)**(i+1) + 1) * np.pi / 2.0
        return phi


class TwistDefinition(SyncStateDefinition):
    def __init__(self, system, m):
        if isinstance(system.g.arr, Ring):
            super(TwistDefinition, self).__init__(system)
            self.m = m
        else:
            raise Exception('State definition not compatible with system.')

    def get_phi(self):
        '''The static phases of all oscillators

           The phase of oscillator 0 is set to 0.
        '''
        n = self.sys.g.arr.get_n()
        dphi = 2 * np.pi /float(n) * self.m
        phi = dphi * np.arange(n)
        return phi


class CubicTwistDefinition(SyncStateDefinition):
    def __init__(self, system, mx, my):
        if (isinstance(system.g.arr, PeriodicCubic2D) or
            (isinstance(system.g.arr, OpenCubic2D) and mx == 0 and my == 0)):
            super(CubicTwistDefinition, self).__init__(system)
            self.mx = mx
            self.my = my
        else:
            raise Exception('State definition not compatible with system.')

    def get_phi(self):
        '''The static phases of all oscillators in index representation

           The phase of oscillator 0 is set to 0.
        '''
        nx = self.sys.g.arr.nx
        ny = self.sys.g.arr.ny
        dphix = 2 * np.pi /float(nx) * self.mx
        dphiy = 2 * np.pi /float(ny) * self.my
        phi = np.zeros((ny, nx))

        for iy in range(ny):
            for ix in range(nx):
                phi[iy, ix] = ix * dphix + iy * dphiy

        phi = phi.flatten()
        return phi


class CubicCheckerboardDefinition(SyncStateDefinition):
    def __init__(self, system):
        if isinstance(system.g.arr, OpenCubic2D):
            super(CubicCheckerboardDefinition, self).__init__(system)
        else:
            raise Exception('State definition not compatible with system.')

    def get_phi(self):
        '''The static phases of all oscillators in index representation

           The phase of oscillator 0 is set to 0.
        '''
        nx = self.sys.g.arr.nx
        ny = self.sys.g.arr.ny
        phi = np.zeros((ny, nx))

        for iy in range(ny):
            for ix in range(nx):
                phi[iy, ix] =  0.5 * ((-1)**(ix + iy + 1) + 1) * np.pi / 2.0

        phi = phi.flatten()
        return phi



# #############################################################################
# State factory
# #############################################################################

class SyncStateFactory(object):
    def __init__(self, state_def):
        if (isinstance(state_def, CheckerboardDefinition) or
            isinstance(state_def, TwistDefinition) or
            isinstance(state_def, CubicTwistDefinition) or
            isinstance(state_def, CubicCheckerboardDefinition)):
            self.sys = state_def.sys
            self.state_def = state_def
        else:
            raise Exception('State definition type not supported by SyncStateFactory')

    def get_phi(self):
        return self.state_def.get_phi()

    def get_dphi_matrix(self):
        return self.state_def.get_dphi_matrix()

    def get_coupling_sum(self, omega, k=1):
        ''' The sum of all coupling function interacting with oscillator k'''
        tau = self.sys.g.tau
        h = self.sys.g.func
        dphi = self.get_dphi_matrix()[k, :]
        c = self.sys.g.get_single_site_coupling(k)
        omega = np.asarray(omega)
        if omega.size > 1:
            h_sum = np.zeros(omega.size)
            for i in range(omega.size):
                h_sum[i] = np.sum(c * h(-tau * omega[i] + dphi))
        else:
            h_sum = np.sum(c * h(-tau * omega + dphi))
        return h_sum

    def get_omega(self, k=1, ns=1000):
        # Get parameters
        tau = self.sys.g.tau
        kc = self.sys.g.k
        w = self.sys.pll.w

        # Determine min and max values for coupling sum function
        h_min = self.sys.g.func.min()
        h_max = self.sys.g.func.max()
        c_bar = self.sys.g.get_single_site_coupling(k)
        c_bar_sum = np.sum(c_bar)       # should be 1 for normalized coupling
        h_sum_min = c_bar_sum * h_min
        h_sum_max = c_bar_sum * h_max

        # Determine search interval for s
        s_min = kc * tau * h_sum_min + w * tau
        s_max = kc * tau * h_sum_max + w * tau
        s_min = s_min - 2  # add safety margin
        s_max = s_max + 2
        if s_min < 0:
            s_min = 0.0   # exclude negative frequencies
        s = np.linspace(s_min, s_max, ns)

        # Setup coupling sum function
        if tau != 0:
            h_sum = lambda x: self.get_coupling_sum(x / tau, k=k)
        else:
            h_sum = lambda x: self.get_coupling_sum(0, k=k)

        # Setup implicit equation for s
        f = lambda x: kc * tau * h_sum(x) + w * tau - x

        # Find sign changes as you go along curve
        # Assumes that there are no double sign changes between two values of s
        # A finer sampling interval can be achieved by increasing ns
        i_root = get_sign_changes(f(s))
        if len(i_root) > 0:
            omega = []
            for i in range(len(i_root)):
                # Numerically solve the implicit equation for omega
                s_tmp = optimize.brentq(f, s[i_root[i]], s[i_root[i] + 1])
                omega.append(w + kc * h_sum(s_tmp))
            return omega
        else:
            raise Exception('No global synchronization frequency found.')

    def get_states(self):
        omega = self.get_omega()
        states = []
        for o in omega:
            if isinstance(self.state_def, CheckerboardDefinition):
                states.append(Checkerboard(self.state_def, o))
            elif isinstance(self.state_def, TwistDefinition):
                states.append(Twist(self.state_def, o))
            elif isinstance(self.state_def, CubicCheckerboardDefinition):
                states.append(CubicCheckerboard(self.state_def, o))
            elif isinstance(self.state_def, CubicTwistDefinition):
                states.append(CubicTwist(self.state_def, o))
        return states




# #############################################################################
# State classes
# #############################################################################

class SyncState(object):
    def __init__(self, state_def, omega):
        self.state_def = state_def
        self.sys = state_def.sys
        self.omega = omega

    def get_phi(self):
        return self.state_def.get_phi()

    def get_dphi_matrix(self):
        return self.state_def.get_dphi_matrix()

    def get_stability(self, l0=np.array([1.0, 1.0])):
        funcs = self.get_stability_functions()
        l = []
        for f in funcs:
            l_full = optimize.root(f, l0, tol=1e-14)
            l_num = l_full.x[0] + 1j * l_full.x[1]
            l.append(l_num)
        l = np.array(l)
        i_max = np.argmax(np.real(l))
        return l[i_max]

    def get_stability_functions(self, k=1):
        funcs = []
        b = self.sys.pll.b
        kc = self.sys.g.k
        d = self.get_coupling_derivative_matrix()
        d_sum = np.sum(d[k, :])
        tau = self.sys.g.tau
        e, v = self.get_eigensystem()
        for el in e:
            func_root = lambda l: self._stability_function(l, b, kc, d_sum, tau, el)
            funcs.append(func_root)

        return funcs

    def get_coupling_derivative_matrix(self):
        dphi = self.get_dphi_matrix()
        h = self.sys.g.func
        dhdx = h.get_derivative()
        c = self.sys.g.get_coupling_matrix()
        return c * dhdx(-self.omega * self.sys.g.tau + dphi)

    def get_eigensystem(self, cutoff=1e-6):
        m = self.get_coupling_derivative_matrix()
        e, v = np.linalg.eig(m)

        e_tmp = []
        v_tmp = []
        for ie in range(len(e)):
            dv_max = np.max(np.abs(np.diff(v[:, ie])))
            if dv_max >= cutoff:
                e_tmp.append(e[ie])
                v_tmp.append(v[:, ie])
        e_tmp = np.array(e_tmp)
        v_tmp = np.transpose(np.array(v_tmp))

        return e_tmp, v_tmp

    @staticmethod
    def _stability_function(l_vector, b, kc, d_sum, tau, eig_cx):
        x = np.zeros(2)
        l_cx = l_vector[0] + 1j * l_vector[1]
        y = l_cx * (1 + b * l_cx) + kc * d_sum - kc * np.exp(-l_cx * tau) * eig_cx
        x[0] = np.real(y)
        x[1] = np.imag(y)
        return x


class Checkerboard(SyncState):
    def __init__(self, state_def, omega):
        if isinstance(state_def, CheckerboardDefinition):
            super(Checkerboard, self).__init__(state_def, omega)
        else:
            raise Exception('Non-compatible StateDefinition')


class Twist(SyncState):
    def __init__(self, state_def, omega):
        if isinstance(state_def, TwistDefinition):
            super(Twist, self).__init__(state_def, omega)
        else:
            raise Exception('Non-compatible StateDefinition')


class CubicCheckerboard(SyncState):
    def __init__(self, state_def, omega):
        if isinstance(state_def, CubicCheckerboardDefinition):
            super(CubicCheckerboard, self).__init__(state_def, omega)
        else:
            raise Exception('Non-compatible StateDefinition')


class CubicTwist(SyncState):
    def __init__(self, state_def, omega):
        if isinstance(state_def, CubicTwistDefinition):
            super(CubicTwist, self).__init__(state_def, omega)
        else:
            raise Exception('Non-compatible StateDefinition')