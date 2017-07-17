import datetime
import os

# Required when plot windows should not be displayed
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import synctools3 as st


TOPO_0D_GLOABL = 'global'
TOPO_1D_RING = 'ring'
TOPO_1D_CHAIN = 'chain'
TOPO_2D_CUBIC_OPEN = 'square-open'
TOPO_2D_CUBIC_PERIODIC = 'square-periodic'
TOPO_2D_HEXAGONAL_OPEN = 'hexagon-open'
TOPO_2D_HEXAGONAL_PERIODIC = 'hexagon-periodic'
TOPO_2D_OCTAGONAL_OPEN = 'octagon-open'
TOPO_2D_OCTAGONAL_PERIODIC = 'octagon-periodic'


COUPLING_FUNCTION_COS = 'cos'
COUPLING_FUNCTION_SIN = 'sin'
COUPLING_FUNCTION_TRIANGLE = 'triang'



# #############################################################################



def generate_delay_plot(n, ny, nx, w, k, h, wc, m, mx, my, topology, isRadians=True, filename=None):
    # Setup sweep factory and create state list
    n_points = 250
    if isRadians:
        tau_max = 2.0 / (w / (2 * np.pi))
        f = w / (2 * np.pi)
        fc = wc / (2 * np.pi)
        kc = k  / (2 * np.pi)
    else:
        tau_max = 2.0 / w
        f = w
        fc = wc
        kc = k
    tau = np.linspace(0, tau_max, n_points)
    sf = SweepFactory(n, ny, nx, w, k, tau, h, wc, m, mx, my, topology, 0, isRadians=isRadians)
    fsl = sf.sweep()

    # Create parameter string
    str_para = ''
    str_para += 'k = %i   kx = %i   ky = %i' % (m, mx, my)
    str_para += '\n%s topology' % topology
    str_para += ' n = %i   nx = %i   ny = %i' % (n, nx, ny)
    str_para += '\nF = %.2f Hz   Fc = %.2f Hz   Kc = %.2f Hz' % (f, fc, kc)

    # Create figure
    plt.figure(figsize=(8, 8.2))

    plt.subplot(2, 1, 1)
    plt.title(str_para)
    plt.plot(fsl.get_tau(), fsl.get_omega(), '.')
    plt.grid(True, ls='--')
    plt.xlabel('Delay [s]')
    plt.ylabel('Sync. frequency [rad/s]')
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.axhline(0, color='k')
    plt.plot(fsl.get_tau(), np.real(fsl.get_l()), '.')
    plt.grid(True, ls='--')
    plt.xlabel('Delay [s]')
    plt.ylabel('Stability [rad/s]')
    plt.tight_layout()
    plt.draw()

    # Save figure
    if filename == None:
        dt = datetime.datetime.now()
        str_time = dt.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join('results', 'delay_plot_' + str_time)
    plt.savefig(filename + '.png', dpi=150)
    plt.savefig(filename + '.pdf')

    # Show figure
    plt.show()




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
        self.n = n
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.h = h
        self.m = m
        self.mx = mx
        self.my = my
        self.tsim = tsim
        self.topology = topology
        self.c = c                     # just dummy variable here

        # if parameters provided in rad*Hz
        if isRadians:
            self.w    = w
            self.k    = k
            self.wc   = wc
        # if parameters provided in Hz, multiply by 2pi, as needed in the phase model
        else:
            self.w    = 2.0*np.pi*w           # here, w = f
            self.k    = 2.0*np.pi*k           # here, k is given in Hz instead rad*Hz
            self.wc   = 2.0*np.pi*wc          # here, wc = fc


        # Identify and store swept variable
        self.key_sweep = self._identify_swept_variable()
        self.values_sweep = self[self.key_sweep]
        self[self.key_sweep] = self.values_sweep[0]


    def _identify_swept_variable(self):
        '''Identify the swept variable

           Returns
           -------
           var_str  :  str
                       name string of the swept variable
        '''
        if type(self.n) is np.ndarray:
            return 'n'
        elif type(self.nx) is np.ndarray:
            return 'nx'
        elif type(self.ny) is np.ndarray:
            return 'ny'
        elif type(self.tau) is np.ndarray:
            return 'tau'
        elif type(self.w) is np.ndarray:
            return 'w'
        elif type(self.k) is np.ndarray:
            return 'k'
        elif type(self.wc) is np.ndarray:
            return 'wc'
        elif type(self.c) is np.ndarray:
            return 'c'
        else:
            return None

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def init_system(self):
        # Initilaize coupling function
        if self.h == COUPLING_FUNCTION_TRIANGLE:
            h_func = st.Triangle(1.0 / (2.0 * np.pi))
        elif self.h == COUPLING_FUNCTION_COS:
            h_func = st.Cos(1.0 / (2.0 * np.pi))
        elif self.h == COUPLING_FUNCTION_COS:
            h_func = st.Sin(1.0 / (2.0 * np.pi))
        else:
            raise Exception('Non-valid coupling function string')

        # Initialize arrangement/ and coupling
        if self.topology == TOPO_1D_CHAIN:
            arr = st.Chain(self.n)
            g = st.NearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
        elif self.topology == TOPO_1D_RING:
            arr = st.Ring(self.n)
            g = st.NearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
        elif self.topology == TOPO_2D_CUBIC_OPEN:
            arr = st.OpenCubic2D(self.nx, self.ny)
            g = st.CubicNearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
        elif self.topology == TOPO_2D_CUBIC_PERIODIC:
            arr = st.PeriodicCubic2D(self.nx, self.ny)
            g = st.CubicNearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
        elif self.topology == TOPO_2D_HEXAGONAL_OPEN:
            arr = st.OpenCubic2D(self.nx, self.ny)
            g = st.CubicHexagonal(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
        elif self.topology == TOPO_2D_HEXAGONAL_PERIODIC:
            arr = st.PeriodicCubic2D(self.nx, self.ny)
            g = st.CubicHexagonal(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
        elif self.topology == TOPO_2D_OCTAGONAL_OPEN:
            arr = st.OpenCubic2D(self.nx, self.ny)
            g = st.CubicOctagonal(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
        elif self.topology == TOPO_2D_OCTAGONAL_PERIODIC:
            arr = st.PeriodicCubic2D(self.nx, self.ny)
            g = st.CubicOctagonal(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)
        else:
            raise Exception('Non-valid topology string')

        # Initialize singel pll
        pll = st.Pll(self.w, self.wc)

        # Initialize system
        pll_sys = st.PllSystem(pll, g)

        return pll_sys

    def get_states(self, pll_sys):
        # 1d twist state
        if self.topology == TOPO_1D_RING:
            state_def = st.TwistDefinition(pll_sys, self.m)
        # 1d global sync state for non-periodic boundray conditions
        elif self.topology == TOPO_1D_CHAIN and self.m == 0:
            state_def = st.TwistDefinition(pll_sys, 0)
        # 1d Checkerboard states for non-periodic boundray conditions and m > 0
        elif self.topology == TOPO_1D_CHAIN:
            state_def = st.CheckerboardDefinition(pll_sys)
        # Global sync state for open 2d cubic lattice
        elif (self.topology == TOPO_2D_CUBIC_OPEN or self.topology == TOPO_2D_HEXAGONAL_OPEN or self.topology == TOPO_2D_OCTAGONAL_OPEN) and self.mx == 0 and self.my == 0:
            state_def = st.CubicTwistDefinition(pll_sys, 0, 0)
        # Checkerboard state for open cubic 2d lattice
        elif (self.topology == TOPO_2D_CUBIC_OPEN or self.topology == TOPO_2D_HEXAGONAL_OPEN or self.topology == TOPO_2D_OCTAGONAL_OPEN) and (self.mx > 0 or self.my > 0):
            state_def = st.CubicCheckerboardDefinition(pll_sys)
        # Twist states for periodic cubic 2d lattice
        elif (self.topology == TOPO_2D_CUBIC_PERIODIC or self.topology == TOPO_2D_HEXAGONAL_PERIODIC or self.topology == TOPO_2D_OCTAGONAL_PERIODIC):
            state_def = st.CubicTwistDefinition(pll_sys, self.mx, self.my)
        else:
            raise Exception('Interface does not support topology yet.')

        return state_def.get_states()

    def sweep(self):
        '''Performs sweep

           Determines the possible globally synchronized states, their angular frequencies and their linear stability

           Returns
           -------
           fsl  :  FlatStateList
                   flat list of the possible states
        '''
        # Set up sweep loop
        fsl = FlatStateList(sweep_factory=self)
        for i in range(len(self.values_sweep)):
            msg_str = 'Sweep value: %.3e' % self.values_sweep[i]
            print msg_str + '\r',

            # Set new value for sweep variable
            self[self.key_sweep] = self.values_sweep[i]

            # Construct system
            pll_sys = self.init_system()

            # Get states
            s = self.get_states(pll_sys)
            fsl.add_states(s)

        return fsl





# ##############################################################################

class FlatStateList(object):
    '''Flat list of TwistStates'''
    def __init__(self, tsim=0.0, sweep_factory=None):
        self.states = []
        self.n = 0
        self.tsim = tsim
        self.sweep_factory = sweep_factory

    def add_states(self, s):
        '''Adds a single or a list of twist states to the list

           Parameters
           ----------
           s : Twist
           State or list of TwistStates
               state or list of states that should be added
        '''
        if isinstance(s, st.SyncState):
            self.states.append(s)
            self.n = len(self.states)
        elif isinstance(s, list):
            for el in s:
                self.states.append(el)
            self.n = len(self.states)
        else:
            raise Exception('Non-valid object for storage in FlatStateList')

    def get_n(self):
        '''Returns an array of the number of oscillators of the states in the list'''
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].sys.g.arr.get_n()
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
                x[i] = s * self.states[i].sys.pll.w
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
                x[i] = s * self.states[i].sys.g.k
            return x
        else:
            return None

    def get_tau(self):
        '''Returns an array of the delay times of the states in the list'''
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                x[i] = self.states[i].sys.g.tau
            return x
        else:
            return None

    def get_m(self):
        '''Returns an array of the twist numbers of the states in the list'''
        return self.get_mx()

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
                x[i] = self.states[i].get_stability()
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
                x[i] = s * self.states[i].sys.pll.wc
            return x
        else:
            return None

    def get_tsim(self):
        '''Returns an array of simulation time'''
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                re_lambda = np.real(self.states[i].get_stability())
                x[i] = 25.0 / np.abs(re_lambda)
            return x
        else:
            return None

    def get_nx(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                s = self.states[i]
                if isinstance(s.sys.g.arr, st.Linear):
                    x[i] = s.sys.g.arr.get_n()
                elif isinstance(s.sys.g.arr, st.Cubic2D):
                    x[i] = s.sys.g.arr.nx
                else:
                    raise Exception('Topology not yet supported')
            return x
        else:
            return None

    def get_ny(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                s = self.states[i]
                if isinstance(s.sys.g.arr, st.Linear):
                    x[i] = 1
                elif isinstance(s.sys.g.arr, st.Cubic2D):
                    x[i] = s.sys.g.arr.ny
                else:
                    raise Exception('Topology not yet supported')
            return x
        else:
            return None

    def get_mx(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                s = self.states[i]
                if isinstance(s, st.Twist):
                    x[i] = s.state_def.m
                elif isinstance(s, st.Checkerboard):
                    #x[i] = s.sys.g.arr.get_n() / 2
                    x[i] = self.sweep_factory.m     # Required by Lucas' code
                elif isinstance(s, st.CubicTwist):
                    x[i] = s.state_def.mx
                elif isinstance(s, st.CubicCheckerboard):
                    #x[i] = s.sys.g.arr.nx / 2
                    x[i] = self.sweep_factory.mx    # Required by Lucas' code
                else:
                   raise Exception('State not supported so far.')
            return x
        else:
            return None

    def get_my(self):
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                s = self.states[i]
                if isinstance(s, st.Twist):
                    x[i] = -999
                elif isinstance(s, st.Checkerboard):
                    x[i] = -999
                elif isinstance(s, st.CubicTwist):
                    x[i] = s.state_def.my
                elif isinstance(s, st.CubicCheckerboard):
                    # x[i] = s.sys.g.arr.ny / 2
                    x[i] = self.sweep_factory.my        # Required by Lucas' code
                else:
                    raise Exception('State not supported so far.')
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
        if self.n > 0:
            x = np.zeros((self.n, 14))
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
            x[:, 10] = self.get_nx()
            x[:, 11] = self.get_ny()
            x[:, 12] = self.get_mx()
            x[:, 13] = self.get_my()
            return x
        else:
            return None


