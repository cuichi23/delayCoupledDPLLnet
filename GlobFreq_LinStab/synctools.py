# ##############################################################################
# Object-oriented parameter sweep
# Version 1: Initial version
# Author: Daniel Platz
# ##############################################################################
import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize

import networkx

# ##############################################################################

class Dcosdt(object):
    '''Periodic sine wave vertically centered around 0'''
    def __init__(self, freq):
        self.freq = freq

    def __call__(self, t):
        return ( -1.0 * np.sin(2 * np.pi * self.freq * t) )

class Dsindt(object):
    '''Periodic sine wave vertically centered around 0'''
    def __init__(self, freq):
        self.freq = freq

    def __call__(self, t):
        return ( 1.0 * np.cos(2 * np.pi * self.freq * t) )

class Square(object):
    '''Periodic square wave vertically centered around 0'''
    def __init__(self, freq, amp):
        self.freq = freq
        self.amp = amp

    def __call__(self, t):
        return self.amp * signal.square(2 * np.pi * self.freq * t, duty=0.5)

class Triangle(object):
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

class Cos(object):
    ''' Periodic triangle signal vertically centered around 0'''
    def __init__(self, freq):
        self.freq = freq

    def __call__(self, t):
        return np.cos(2 * np.pi * self.freq * t)

    def get_derivative(self):
        return Dcosdt(self.freq)

    def max(self):
        return 1.0

    def min(self):
        return -1.0

class Sin(object):
    ''' Periodic triangle signal vertically centered around 0'''
    def __init__(self, freq):
        self.freq = freq

    def __call__(self, t):
        return np.sin(2 * np.pi * self.freq * t)

    def get_derivative(self):
        return Dsindt(self.freq)

    def max(self):
        return 1.0

    def min(self):
        return -1.0

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
    '''Compute a parametric curve of the global synchronization frequency as a function of the delay.

      Parameters
      ----------
      n : int
          number of oscillators
      w : float
          intrinsic angular frequency of the oscillators
      k : float
          coupling constant
      h : callable
          coupling function
      m : int
          twist number
      s_min : float
              minimum value of the curve parameter
      s_max : float
              minimum value of the curve parameter
      ds : float
           discritization step of the curve parameter

      Returns
      -------
      tau : np.ndarray
            delay
      omega : np.ndarray
              global angular frequency of the synchronized system
      s : np.ndarray
          used curve parameter values
   '''
    phi_m = (2 * np.pi * m) / n
    n_s = np.int(np.rint((s_max - s_min) / ds))
    s = np.linspace(s_min, s_max, n_s)
    omega = w + (k / 2.0) * (h(-s + phi_m) + h(-s - phi_m))
    tau  = s / omega

    return tau, omega, s


def get_omega_implicit(n, w, k, tau, h, m):
    '''Computes the global synchronization frequency for a given delay.

      Based in nonlinear implicit equation of global synchronization frequency

      Parameters
      ----------
      n : int
          number of oscillators
      w : float
          intrinsic angular frequency of the oscillators
      k : float
          coupling constant
      tau : float
            delay
      h : callable
          coupling function
      m : int
          twist number

      Returns
      -------
      omega : list/None
              list of possible gloabally synchronized states for given parameters. None if there is no possible state
   '''
    phi_m = (2 * np.pi * m) / n
    func = lambda s: -s + 0.5 * k * tau * (h(-s + phi_m) + h(-s - phi_m)) + w * tau

    # Determine values of the curve parameter under the assumption of a bound coupling functions
    h_min = 2 * h.min()
    h_max = 2 * h.max()
    s_min = h_min + w * tau
    s_max = h_max + w * tau
    s = np.linspace(s_min, s_max, 10000)

    # Find sign changes as you go along curve
    # Assumes that there are no double sign changes between two values of s
    i_root = get_sign_changes(func(s))
    if len(i_root) > 0:
        omega = []
        for ir in range(len(i_root)):
            # Numerically solve the implicit equation for omega
            s_tmp = optimize.brentq(func, s[i_root[ir]], s[i_root[ir] + 1])
            omega.append(w + 0.5 * k * (h(-s_tmp + phi_m) + h(-s_tmp - phi_m)))
        return omega
    else:
        return None

def chooseTwistNumbers(nx, ny):   												# ask user-input for delay
    a_true = True
    while a_true:
        # get user input on number of oscis in the network
        k1 = raw_input('\nPlease specify the first (x-direction) twist number for 2d m1-m2-twist solutions [integer] in [0, ..., %d] [dimless]: ' %(nx-1))
        k2 = raw_input('\nPlease specify the second (y-direction) twist number for 2d m1-m2-twist solutions [integer] in [0, ..., %d] [dimless]: ' %(ny-1))
        if ( int(k1)>=0 and int(k2)>=0 ):
            break
        else:
            print('Please provide input as an [integer] in [0, %d]!' %(N-1))

    return int(k1), int(k2)

def calcTopoMatrix(n, nx, ny, w, k, h, m, tau, omega, wc, topology):
    # Dependent parameters
    b = 1.0 / wc
    dhdt = h.get_derivative()

    # Construct coupling matrix and compute its eigensystem
    if topology == 'global':
        G = networkx.complete_graph(n)
        d = np.zeros((n, n))                                                    # prepare coupling topology matrix with 0 and 1, then normalize
        for ir in range(n):                                                     # iterate and fill
            ir_neigh = G.neighbors(ir)
            d[ir, ir_neigh] = 1
            d[ir, :] = d[ir, :]  / np.sum(d[ir, :])
        e_mat = d

        Should we put K into the matrix, check consistency for all cases

        print('Global coupling does not support m-twist solutions! Careful here, recheck.')

    elif ( topology == 'hexagon' or topology == 'octagon' ):

        N = np.sqrt(n)
		if N.is_integer():
			N = int(N)
		else:
			raise ValueError('Npll is not valid: sqrt(N) is not an integer')
        elif topology == 'hexagon':
            print('\nOpen boundary conditions in this case, extend code... add part with edges that span "around"!\n')
            G=networkx.grid_2d_graph(N,N)
            for n in G:
                x,y=n
                if x>0 and y>0:
                    G.add_edge(n,(x-1,y-1))
                if x<N-1 and y<N-1:
                    G.add_edge(n,(x+1,y+1))
        elif topology == 'octagon':
            print('\nOpen boundary conditions in this case, extend code... add part with edges that span "around"!\n')
            G=networkx.grid_2d_graph(N,N)
            for n in G:
                x,y=n
                if x>0 and y>0:
                    G.add_edge(n,(x-1,y-1))
                if x<N-1 and y<N-1:
                    G.add_edge(n,(x+1,y+1))
                if x<N-1 and y>0:
                    G.add_edge(n,(x+1,y-1))
                if x<N-1 and y>0:
                    G.add_edge(n,(x+1,y-1))
                if x>0 and y<N-1:
                    G.add_edge(n,(x-1,y+1))
        # matrix components are numbered from 1 to N^2, not for kl, each 1 to N
        G = networkx.convert_node_labels_to_integers(G, ordering='sorted')

    elif ( topology == 'ring' or 'chain' ):
        # Determine help variables
        alpha_minus = 0.5 * k * dhdt(-omega * tau + phi_m)                      # factor 0.5, since 2 neighbors in 1d
        alpha_plus  = 0.5 * k * dhdt(-omega * tau - phi_m)
        if topology == 'ring':
            ''' 1d ring topology, periodic boundary conditions '''
            e_mat = np.zeros((n, n))
            e_mat[0, -1] = alpha_plus
            e_mat[0, 1] = alpha_minus
            for ik in range(1, n - 1):
                e_mat[ik, ik - 1] = alpha_plus
                e_mat[ik, ik + 1] = alpha_minus
            e_mat[-1, 0] = alpha_minus
            e_mat[-1, -2] = alpha_plus

        elif topology == 'chain':
            ''' 1d chain topology, open boundary conditions '''
            e_mat = np.zeros((n, n))
            e_mat[0, -1] = 0.0
            e_mat[0, 1] = alpha_minus
            for ik in range(1, n - 1):
                e_mat[ik, ik - 1] = alpha_plus
                e_mat[ik, ik + 1] = alpha_minus
            e_mat[-1, 0] = 0.0
            e_mat[-1, -2] = alpha_plus

    elif ( topology == 'square-open' or topology == 'square-periodic' ):

        if topology == 'square-open':
            G=networkx.grid_2d_graph(nx, ny)
            delta_phase_chequer = np.pi
            print('Check chequerboard case in synctools again!')
        elif topology == 'square-periodic':
            G=networkx.grid_2d_graph(nx, ny, periodic=True)                     # for periodic boundary conditions:
            mx, my = chooseTwistNumbers(nx, ny)

        G = networkx.convert_node_labels_to_integers(G, ordering='sorted')
        ''' Normalization '''
        d = np.zeros((n, n))                                                    # prepare coupling topology matrix with 0 and 1, then normalize
        for ir in range(n):                                                     # iterate and fill
            ir_neigh = G.neighbors(ir)
            d[ir, ir_neigh] = 1
            d[ir, :] = d[ir, :]  / np.sum(d[ir, :])

        # Determine help variables
        # alpha_minus = 0.5 * np.pi * k * dhdt(-omega * tau + phi_m)
        delta_phi_mx = (2.0 * np.pi * mx) / nx
        delta_phi_my = (2.0 * np.pi * my) / ny

        # if type(h) == Triangle:
        #     prefactor = 2.0 / np.pi
        # else:
        #     prefactor = 1.0
        prefactor = 1.0
        a = np.zeros((n, n))                                                    # prepare coupling matrix that includes the phase-differences of m-twist solutions as property of the topology
        for ir in range(n):                                                     # iterate and fill
            for ic in range(n):
                a[ir, ic] = k * dhdt( -omega * tau + delta_phi_mx *( np.mod(ic,float(nx))-np.mod(ir,float(nx)) ) +
                                                     delta_phi_my *( np.floor(ic/float(nx))-np.floor(ir/float(nx)) ) )

        e_mat = d * a                                                           # element-wise multiplication

    return e_mat

def get_stability(n, nx, ny, w, k, h, m, tau, omega, wc, topology):
    '''Linear stability analysis of globally synchronized state.

       The computation is based on finding the roots of the two-dimensional characteristic equation.
       For each eigenvalue of the coupling matrix there is a different characteristic equation. Here, only
       the complex exponent with the largest real value of all characteristic equations is returned. Moreover,
       the characteristic equation associated with the eigenvector (a, a, a, ...) is excluded since it
       corresponds to a global phase shift of all oscillators and does not affect synchronization.

      Parameters
      ----------
      n : int
          number of oscillators
      w : float
          intrinsic angular frequency of the oscillators
      k : float
          coupling constant
      tau : float
            delay
      h : callable
          coupling function
      m : int
          twist number
      wc : float
          angular cut-off frequency of the low-pass filter

      Returns
      -------
      lambda_nu : complex
                  the complex linear stability analysis exponent with the biggest real value
    '''

    e_mat = calcTopoMatrix(n, nx, ny, w, k, h, m, tau, omega, wc, topology)

    # obtain eigenvectors and eigenvalues
    em, vm = np.linalg.eig(e_mat)
    # print('EIGENVALUES:', em)

    # Solve characteristic equation for each eigenvalue
    lambda_nu = []
    for inu in range(len(em)):
        nu = em[inu]
        #print 'tau = %.3f: %.6f + 1j * %.6f' % (tau, np.real(nu), np.imag(nu))
        def func(l):
            mu = np.real(nu)
            gamma = np.imag(nu)
            x = np.zeros(2)

            case differentiation 1d and 2d!

            x[0] = b * l[0]**2 - b * l[1]**2 + l[0] + 0.5 * (alpha_plus + alpha_minus) - 0.5 * mu * np.exp(-l[0] * tau) * np.cos(l[1] * tau) - 0.5 * gamma * np.exp(-l[0] * tau) * np.sin(l[1] * tau)
            x[1] = 2 * b * l[0] * l[1] + l[1] + 0.5 * mu * np.exp(-l[0] * tau) * np.sin(l[1] * tau) - 0.5 * gamma * np.exp(-l[0] * tau) * np.cos(l[1] * tau)
            return x

        l_opt = optimize.root(func, np.array([1.0, 1.0]), tol=1e-14)
        l_tmp = l_opt.x[0] + 1j * l_opt.x[1]

        # Ignore solution for the eigenvector (a, a, a, ...)
        if np.max(np.abs(np.diff(vm[:, inu]))) >= 1e-9:
            lambda_nu.append(l_tmp)

    lambda_nu = np.array(lambda_nu)
    return lambda_nu[np.argmax(np.real(lambda_nu))]



# ##############################################################################

class PllSystem(object):
    '''System of coupled PLLs

       Attributes
       ----------
       n : int
           number of oscillators
       w : float
           intrinsic angular frequency of the oscillators
       k : float
           coupling constant
       tau : float
             delay time
       h : callable
           coupling function
       wc : float
            low-pass filter (angular) cut-off frequency
       topology : string
            determines the coupling topology of the network
    '''
    def __init__(self, n, w, k, tau, h, wc, topology):
        self.n        = n
        self.w        = w
        self.k        = k
        self.tau      = tau
        self.h        = h
        self.wc       = wc
        self.topology = topology

    def get_twist_state(self, m):
        '''Determine the possible states of global synchronization for a specific m twist

           Parameters
           ----------
           m : int
               twist numer

           Returns
           -------
           s : list of twist states or None
        '''
        o = get_omega_implicit(self.n, self.w, self.k, self.tau, self.h, m)
        if o != None:
            s = []
            for el in o:
                l = get_stability(self.n, self.w, self.k, self.h, m, self.tau, el, self.wc, self.topology)
                s.append(TwistState(self, m, el, l))
            return s
        else:
            return None



class TwistState(object):
    '''Globally synchronized twist states

       Attributes
       ----------
       system : PllSystem
                underlying coupled pll system
       m : int
           twist number
       omega : float
               global (angular) frequency
       l : complex
           complex linear stability analysis exponent
    '''
    def __init__(self, system, m, omega, l):
        self.m = m
        self.omega = omega
        self.l = l
        self.system = system


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
            x[:, 9] = -25.0/x[:, 7]                                             #self.get_tsim()
            return x
        else:
            return None



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
    def __init__(self, n, w, k, tau, h, wc, m, topology, tsim=0.0, isRadians=True):
        if isRadians:                                                           # if parameters provided in rad*Hz
            self.n = n
            self.w = w
            self.k = k
            self.tau = tau
            self.h = h
            self.wc = wc
            self.m = m
            self.tsim = tsim
            self.topology = topology
        else:                                                                   # if parameters provided in Hz, multiply by 2pi, as needed in the phase model
            self.n = n
            self.w = 2.0*np.pi*w                                                # here, w = f
            self.k = 2.0*np.pi*k                                                # here, k is given in Hz instead rad*Hz
            print('K:', k)
            self.tau = tau
            self.h = h
            self.wc = 2.0*np.pi*wc                                              # here, wc = fc
            self.m = m
            self.tsim = tsim
            self.topology = topology

    def _identify_swept_variable(self):
        '''Identify the swept variable

           Returns
           -------
           var_str  :  str
                       name string of the swept variable
        '''
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
        elif type(self.topology) is list:
            return 'topology'
        else:
            return None

    def __getitem__(self, key):
        return self.__dict__[key]

    def sweep(self):
        '''Performs sweep

           Determines the possible globally synchronized states, their angular frequencies and their linear stability

           Returns
           -------
           fsl  :  FlatStateList
                   flat list of the possible states
        '''
        fsl = FlatStateList(tsim=self.tsim)
        key_sweep = self._identify_swept_variable()
        par_sweep = self[key_sweep]
        n_sweep = len(par_sweep)
        key_sys = ['n', 'w', 'k', 'tau', 'h', 'wc', 'topology']
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
