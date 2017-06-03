# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:36:06 2017

@author: platz
"""
import numpy as np
import scipy.optimize as optimize
import scipy.signal as signal


STATE_TWIST = 0



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
        return 2

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



class Graph(object):
    def __init__(self, arrangement, function, strength, delay, hasNormalizedCoupling):
        self.arr = arrangement
        self.func = function
        self.k = strength
        self.tau = delay
        self.hasNormalizedCoupling = hasNormalizedCoupling

    def get_single_site_coupling(self, i):
        raise NotImplementedError

    def get_single_site_coupling_matrix(self, yx):
        raise NotImplementedError

    def get_coupling_matrix(self):
        raise NotImplementedError



class GraphLinear(Graph):
    pass



class NearestNeighbor(GraphLinear):
    def __init__(self, arrangement, function, strength, delay, hasNormalizedCoupling):
        super(NearestNeighbor, self).__init__(arrangement, function, strength, delay, hasNormalizedCoupling)
        self.d = np.array([-1, 1])

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


# #############################################################################

class Pll(object):
    def __init__(self, w, wc):
        self.w = w
        self.wc =wc
        self.b = 1.0 / wc


# #############################################################################








class PllSystem(object):
    def __init__(self, pll, graph):
        self.pll = pll
        self.g = graph

#    def get_states(self, state_type):
#        if state_type == STATE_TWIST:
#            pass





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


class State(object):
    def __init__(self, system):
        self.sys = system

    @classmethod
    def get_states(cls, sys):
        raise NotImplementedError


class LinearState(State):
    pass


class Twist(LinearState):
    def __init__(self, system, m, omega):
        super(Twist, self).__init__(system)
        self.m = m
        self.omega = omega
        self.sys = system
        n = self.sys.g.arr.get_n()
        self.dphi = Twist.get_dphi(n, m)


    @staticmethod
    def get_states(sys, m, k=1):
        n = sys.g.arr.get_n()
        w = sys.pll.w
        k = sys.g.k
        tau = sys.g.tau
        c = sys.g.get_single_site_coupling(k)
        n = sys.g.arr.get_n()
        h = sys.g.func

        f_sum = lambda s: Twist.get_coupling_sum(c, n, m, h, s, k=1)
        omega = Twist.get_omega(w, k, tau, f_sum, ns=10000)

        states = []
        for o in omega:
            states.append(Twist(sys, m, o))
        return states


    @staticmethod
    def get_dphi(n, m):
        return 2 * np.pi /float(n) * m


    def get_dphi_matrix(self):
        n = self.sys.g.arr.get_n()
        dphi = self.get_dphi(n, self.m)
        dphi_mat = np.zeros((n, n))
        for ik in range(n):
            for il in range(n):
                dphi_mat[ik, il] = il * dphi - ik * dphi
        return dphi_mat


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


    def get_coupling_derivative_sum(self, k=0):
        d = self.get_coupling_derivative_matrix()
        return np.sum(d[k, :])

    @staticmethod
    def _stability_function(l, b, kc, d_sum, tau, alpha, beta):
        x = np.zeros(2)
        zeta = alpha + 1j * beta
        l_cx = l[0] + 1j * l[1]
        y = l_cx * (1 + b * l_cx) + kc * d_sum - np.exp(-l_cx * tau) * zeta
        x[0] = np.real(y)
        x[1] = np.imag(y)
        #x[0] = l[0] + b * l[0]**2 - b * l[1]**2 + kc * d_sum - kc * np.exp(-l[0] * tau) * (np.cos(l[1] * tau) * alpha + np.sin(l[1] * tau) * beta)
        #x[1] = l[1] + 2 * b * l[0] * l[1] - kc * np.exp(-l[0] * tau) * (np.cos(l[1] * tau) * beta - np.sin(l[1] * tau) * alpha)
        return x

    def get_stability_functions(self):
        funcs = []
        b = self.sys.pll.b
        kc = self.sys.g.k
        d_sum = self.get_coupling_derivative_sum()
        tau = self.sys.g.tau
        e, v = self.get_eigensystem()
        for el in e:
            alpha = np.real(el)
            beta = np.imag(el)
            func_root = lambda l: self._stability_function(l, b, kc, d_sum, tau, alpha, beta)
            funcs.append(func_root)

        return funcs

    def get_stability(self, l0=np.array([1.0, 1.0])):
        funcs = self.get_stability_functions()
        l = []
        for f in funcs:
            l_full = optimize.root(f, l0, tol=1e-14)
            l_num = l_full.x[0] + 1j * l_full.x[1]
            l.append(l_num)
        l = np.array(l)
        i_max = np.argmax(np.real(l))
        return np.real(l[i_max]), np.imag(l[i_max])


    @staticmethod
    def get_coupling_sum(c, n, m, h, s, k=0):
        dphi = Twist.get_dphi(n, m)
        dphi = dphi * (np.arange(n) - k)
        s = np.asarray(s)
        if s.size > 1:
            c_sum = np.zeros(s.size)
            for i in range(s.size):
                c_sum[i] = np.sum(c * h(-s[i] + dphi))
        else:
            c_sum = np.sum(c * h(-s + dphi))
        return c_sum


    @staticmethod
    def get_omega(w, k, tau, f_sum, ns=1000):
        # Setup implicit equation for s
        f = lambda s: k * tau * f_sum(s) + w * tau - s

        # Determine search interval for s
        # Assumption c can only vary between -1 and +1
        s_min = (w - k) * tau
        s_max = (w + k) * tau
        s = np.linspace(s_min - 2, s_max + 2, ns)   # safty margin added

        # Find sign changes as you go along curve
        # Assumes that there are no double sign changes between two values of s
        # A finer sampling interval can be achieved by increasing ns
        i_root = get_sign_changes(f(s))
        if len(i_root) > 0:
            omega = []
            for ir in range(len(i_root)):
                # Numerically solve the implicit equation for omega
                s_tmp = optimize.brentq(f, s[i_root[ir]], s[i_root[ir] + 1])
                omega.append(w + k * f_sum(s_tmp))
            return omega
        else:
            return None


# #############################################################################




class SweepFactory(object):
    '''Sweeps a system parameters of a coupled PLL system

       One of the class attributes should be given as a np.ndarray. This will be the swept parameter

       Attributes
       ----------
       n : int/np.ndarray
           number of oscillators^
       w : float/np.ndarray
           intrinsic angular frequency
       k : float/np.ndarray
           coupling constant
       tau : float/np.ndarray
             delay
       h : callable/list of callables
           coupling function
       wc : float/np.ndarray
            (angular) cut-off frequency of low-pass filter
       m : int
           twist number
       tsim : float
              simulation time
    '''
    def __init__(self, n, ny, nx, w, k, tau, h, wc, m, mx, my, topology, c, tsim=0.0, isRadians=True):
        if isRadians:                                                           # if parameters provided in rad*Hz
            self.n    = n
            self.nx   = nx
            self.ny   = ny
            self.w    = w
            self.k    = k
            self.tau  = tau
            self.h    = h
            self.wc   = wc
            self.m    = m
            self.mx   = mx
            self.my   = my
            self.tsim = tsim
            self.topology = topology
            self.c    = c                                                       # just dummy variable here
        else:                                                                   # if parameters provided in Hz, multiply by 2pi, as needed in the phase model
            self.n    = n
            self.nx   = nx
            self.ny   = ny
            self.w    = 2.0*np.pi*w                                             # here, w = f
            self.k    = 2.0*np.pi*k                                             # here, k is given in Hz instead rad*Hz
            print('in SweepFactory, K in [rad*Hz] and [Hz]:', self.k, self.k/(2.0*np.pi))
            self.tau  = tau
            self.h    = h
            self.wc   = 2.0*np.pi*wc                                            # here, wc = fc
            self.m    = m
            self.mx   = mx
            self.my   = my
            self.tsim = tsim
            self.topology = topology
            self.c    = c                                                       # just dummy variable here

    def _identify_swept_variable(self):
        '''Identify the swept variable

           Returns
           -------
           var_str  :  str
                       name string of the swept variable
        '''
        if type(self.w) is np.ndarray:
            return 'w'
        elif type(self.k) is np.ndarray:
            return 'k'
        elif type(self.tau) is np.ndarray:
            return 'tau'
        elif type(self.wc) is np.ndarray:
            return 'wc'
        elif type(self.c) is np.ndarray:
            return 'c'
        else:
            return None

    def __getitem__(self, key):
        return self.__dict__[key]

    @staticmethod
    def init_system(self, topology, n, h, k, tau, w, wc):
        # Initialize arrangement/geometry
            if self.topology == 'TOPO_CHAIN':
                arr = Chain(n)
            elif self.topology == 'TOPO_RING':
                arr = Ring(n)

            # Initialize coupling
            g = NearestNeighbor(arr, h, k, tau, hasNormalizedCoupling=True)

            # Initialize singel pll
            pll = Pll(w, wc)

            # Initialize system
            sys = PllSystem(pll, g)

            return sys



    def sweep(self):
        '''Performs sweep

           Determines the possible globally synchronized states, their angular frequencies and their linear stability

           Returns
           -------
           fsl  :  FlatStateList
                   flat list of the possible states
        '''
        #fsl = FlatStateList(tsim=self.tsim)

        # Identify swept parameter
        key_sweep = self._identify_swept_variable()
        par_sweep = self[key_sweep]
        if not key_sweep=='c':
            n_sweep = len(par_sweep)
        else:
            n_sweep = 1

        fsl = FlatStateList()
        key_sys = ['n', 'w', 'k', 'tau', 'h', 'wc', 'topology', 'nx', 'ny', 'c'] # ['topology', 'n', 'h', 'k', 'tau', 'w', 'wc']
        for i in range(n_sweep):
            args = []
            for key in key_sys:
                if key == key_sweep:
                    args.append(self[key][i])
                else:
                    args.append(self[key])
            sys = self.init_system(*args)

            s = Twist.get_states(sys, self.m)
            fsl.add_states(s)
        return fsl



# ##############################################################################

class FlatStateList(object):
    '''Flat list of TwistStates'''
    def __init__(self, tsim=0.0):
        self.states = []
        self.n = 0
        self.tsim = tsim

    def add_states(self, s):
        '''Adds a single or a list of twist states to the list

           Parameters
           ----------
           s : TwistState or list of TwistStates
               state or list of states that should be added
        '''
        if type(s) is TwistState:
            self.states.append(s)
            self.n = len(self.states)
        elif type(s) is list:
            for el in s:
                self.states.append(el)
            self.n = len(self.states)


    def get_n(self):
        '''Returns an array of the number of oscillators of the states in the list'''
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].system.n
            return x
        else:
            return None


    def get_w(self, isRadians=True):
        '''Returns an array of the intrinsic frequencies of oscillators of the states in the list

           Parameters
           ----------
           isRadians : bool
                       frequency is given in radians if True, otherwise in Hertz
        '''
        if isRadians:
            s = 1.0
        else:
            s = 1.0 / (2 * np.pi)

        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = s * self.states[i].system.w
            return x
        else:
            return None

    def get_k(self, isRadians=True):
        '''Returns an array of the coupling constants of the states in the list

           Parameters
           ----------
           isRadians : bool
                       frequency is given in radians if True, otherwise in Hertz
        '''
        if isRadians:
            s = 1.0
        else:
            s = 1.0 / (2 * np.pi)

        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = s * self.states[i].system.k
            return x
        else:
            return None

    def get_tau(self):
        '''Returns an array of the delay times of the states in the list'''
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].system.tau
            return x
        else:
            return None

    def get_h(self):
        '''Returns a list of the coupling functions of the states in the list'''
        if self.n > 0:
            x = []
            for i in range(self.n):
                x.append(self.states[i].system.h)
            return x
        else:
            return None

    def get_m(self):
        '''Returns an array of the twist numbers of the states in the list'''
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].m
            return x
        else:
            return None

    def get_omega(self, isRadians=True):
        '''Returns an array of the global synchronization frequencies of the states in the list

           Parameters
           ----------
           isRadians : bool
                       frequency is given in radians if True, otherwise in Hertz
        '''
        if isRadians:
            s = 1.0
        else:
            s = 1.0 / (2 * np.pi)

        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = s * self.states[i].omega
            return x
        else:
            return None

    def get_l(self):
        '''Returns an array of the complex linear stability exponent of the states in the list'''
        if self.n > 0:
            x = np.zeros(self.n, dtype=np.complex)
            for i in range(self.n):
                x[i] = self.states[i].l
            return x
        else:
            return None

    def get_wc(self, isRadians=True):
        '''Returns the low-pass filter cut-off frequency of the states in the list

           Parameters
           ----------
           isRadians : bool
                       frequency is given in radians if True, otherwise in Hertz
        '''
        if isRadians:
            s = 1.0
        else:
            s = 1.0 / (2 * np.pi)

        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = s * self.states[i].system.wc
            return x
        else:
            return None

    def get_tsim(self):
        '''Returns an array of simulation time'''
        if self.n > 0:
            x = self.tsim * np.ones(self.n)
            return x
        else:
            return None

    def get_parameter_matrix(self, isRadians=True):
        '''Returns a matrix of the numeric parameters the states in the list

           Parameters
           ----------
           isRadians : bool
                       frequency is given in radians if True, otherwise in Hertz
        '''
        if isRadians:
            s = 1.0
        else:
            s = 1.0 / (2 * np.pi)
        if self.n > 0:
            x = np.zeros((self.n, 10))
            x[:, 0] = self.get_n()
            x[:, 1] = self.get_w(isRadians=isRadians)
            x[:, 2] = self.get_k(isRadians=isRadians)
            x[:, 3] = self.get_wc(isRadians=isRadians)
            x[:, 4] = self.get_tau()
            x[:, 5] = self.get_m()
            x[:, 6] = self.get_omega(isRadians=isRadians)
            x[:, 7] = np.real(self.get_l())
            x[:, 8] = np.imag(self.get_l())
            x[:, 9] = self.get_tsim()
            return x
        else:
            return None
