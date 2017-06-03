import numpy as np
import synctools

# System parameters
n = 3
w = 1.0
k = 0.25
wc = 1 * w
m = 1
tau = np.linspace(0, 2, 250)
h = synctools.Triangle(1.0 / (2 * np.pi))
tsim = 0.1234
topology = 'ring'


# Create a pll system and determine the possible m-twist states
pll = synctools.PllSystem(n, w, k, tau[0], h, wc, 'ring', 0, 0)
s = pll.get_twist_state(m, 0, 0, 'ring')[0]

# Perform a delay sweep
sf = synctools.SweepFactory(n, 0, n, w, k, tau, h, wc, m, 0, 0, topology, tsim=tsim, isRadians=False)
fsl = sf.sweep()

# Extract variables from the sweep
para_mat = fsl.get_parameter_matrix(isRadians=False)
tau2 = fsl.get_tau()
omega2 = fsl.get_omega()
l2 = fsl.get_l()
c_l2 = np.real(l2)
vmin = -np.max(np.abs(c_l2))
vmax = -vmin