import numpy as np

import synctools3 as st


TOPO_0D_GLOABL = 'global'
TOPO_1D_RING = 'ring'
TOPO_1D_CHAIN = 'chain'
TOPO_2D_CUBIC_OPEN = 'square-open'
TOPO_2D_CUBIC_PERIODIC = 'square-periodic'
TOPO_2D_HEXAGONAL_PERIODIC = 'hexgon'
TOPO_2D_OCTAGONAL_PERIODIC = 'octagon'


COUPLING_FUNCTION_COS = 'cos'
COUPLING_FUNCTION_SIN = 'sin'
COUPLING_FUNCTION_TRIANGLE = 'triang'



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

        # Initialize arrangement/geometry
        if self.topology == TOPO_1D_CHAIN:
            arr = st.Chain(self.n)
        elif self.topology == TOPO_1D_RING:
            arr = st.Ring(self.n)
        else:
            raise Exception('Non-valid topology string')

        # Initialize coupling
        g = st.NearestNeighbor(arr, h_func, self.k, self.tau, hasNormalizedCoupling=True)

        # Initialize singel pll
        pll = st.Pll(self.w, self.wc)

        # Initialize system
        pll_sys = st.PllSystem(pll, g)

        return pll_sys


    def get_states(self, pll_sys):
        if self.topology == TOPO_1D_RING:
            state_def = st.TwistDefinition(pll_sys, self.m)
        elif self.topology == TOPO_1D_CHAIN and self.m == 0:
            state_def = st.TwistDefinition(pll_sys, 0)
        elif self.topology == TOPO_1D_CHAIN:
            state_def = st.CheckerboardDefinition(pll_sys)
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
        fsl = FlatStateList()
        for i in range(len(self.values_sweep)):
            print i

            # Construct system
            pll_sys = self.init_system()

            # Get states
            s = self.get_states(pll_sys)
            fsl.add_states(s)

            # Set new value for sweep variable
            self[self.key_sweep] = self.values_sweep[i]

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
        if self.n > 0:
            x = np.zeros(self.n)
            for i in range(self.n):
                s = self.states[i]
                if isinstance(s, st.Twist):
                    x[i] = s.state_def.m
                else:
                    x[i] = 0
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
                else:
                    x[i] = 0
            return x
        else:
            return None



    def get_my(self):
        if self.n > 0:
            x = np.zeros(self.n)
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


