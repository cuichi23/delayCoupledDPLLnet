# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:16:01 2017

@author: platz
"""
import matplotlib.pyplot as plt
import numpy as np

import path_helpers
import mpl_helpers
mpl_helpers.clear_all()

import synctools3 as st



print '\n##########################################################################'
print '5. Compute frequency curve for example parameters'
print '##########################################################################'

n = 3
m = 0
h = st.Triangle(1.0 / (2 * np.pi))
kc = np.pi / 2
w = 2 * np.pi
wc = w
tau = 1.0
b_norm = True


arr = st.Ring(n)
g = st.NearestNeighbor(arr, h, kc, tau, b_norm)
pll= st.Pll(w, wc)
sys = st.PllSystem(pll, g)

tau_sweep = np.linspace(0, 1.7, 200)
omega_states = []
tau_states = []
l_states = []
for t in tau_sweep:
    print t
    sys.g.tau = t
    sd = st.TwistDefinition(sys, m)
    states = sd.get_states()
    for s in states:
        tau_states.append(t)
        omega_states.append(s.omega)
        l_states.append(s.get_stability())


tau_states = np.array(tau_states)
omega_states = np.array(omega_states)
l_states = np.array(l_states)



plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(tau_states, omega_states, 'bo')

plt.subplot(2, 1, 2)
plt.plot(tau_states, l_states, 'bo')


print '##########################################################################'
print 'Clean up'
print '##########################################################################'

mpl_helpers.tight_layout_grid_all()
plt.show()
