import numpy as np

import synctools_interface


n = 3
topology = 'ring'
w = 2 * np.pi
k = np.pi / 2.0
wc = w
h = 'triang'
m = 1
tau = np.linspace(0, 1.7, 5)

ny = 0
nx = 0
mx = 0
my = 0
c = 0


sf = synctools_interface.SweepFactory(n, ny, nx, w, k, tau, h, wc, m, mx, my, topology, c, tsim=0.0, isRadians=True)
fsl = sf.sweep()
para_mat = fsl.get_parameter_matrix()