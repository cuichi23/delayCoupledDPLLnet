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

import synctools2 as st


print '\n##########################################################################'
print '1. Test linear arrangements'
print '##########################################################################'

n = 10
r = st.Ring(n)
c = st.Chain(n)

yx = 5
print r.site_exists(yx)
print c.site_exists(yx)

yx = 15
print r.site_exists(yx)
print c.site_exists(yx)

print r.wrap_coordinates(yx)
print c.wrap_coordinates(yx)




print '\n##########################################################################'
print '2. Test digital coupling function'
print '##########################################################################'


x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
f = st.Triangle(1.0 / (2 * np.pi))
dfdx = f.get_derivative()

print f.max()
print f.min()
print dfdx.max()
print dfdx.min()

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(x, f(x))
plt.xlabel('x')
plt.ylabel('Coupling function f(x)')

plt.subplot(2, 1, 2, sharex=plt.gca())
plt.plot(x, dfdx(x))
plt.xlabel('x')
plt.ylabel('Coupling derivative dfdx(x)')


print '\n##########################################################################'
print '3. Test ring and chain coupling matrices'
print '##########################################################################'

n = 10
ar = st.Ring(n)
ac = st.Chain(n)
tau = 0.0
k = 1.0
b_norm = True
f = st.Triangle(1.0 / (2 * np.pi))
gr = st.NearestNeighbor(ar, f, k, tau, b_norm)
gc = st.NearestNeighbor(ac, f, k, tau, b_norm)

i = 9
vr = gr.get_single_site_coupling(i)
vc = gc.get_single_site_coupling(i)
print vr
print vc


yx = 9
mr = gr.get_single_site_coupling_matrix(yx)
mc = gc.get_single_site_coupling_matrix(yx)
print mr
print mc

mr = gr.get_coupling_matrix()
mc = gc.get_coupling_matrix()
print ''
print mr
print mc

plt.figure(3, figsize=mpl_helpers.SIZE_TUW_HALF)
plt.subplot(2, 1, 1)
plt.title('Ring coupling matrix')
plt.imshow(mr, aspect='auto')
plt.xlabel('Osci index')
plt.ylabel('Osci index')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.title('Chain coupling matrix')
plt.imshow(mc, aspect='auto')
plt.xlabel('Osci index')
plt.ylabel('Osci index')
plt.colorbar()



print '\n##########################################################################'
print '4. Contruct twist object'
print '##########################################################################'


n = 10
ar = st.Ring(n)

f = st.Triangle(1.0 / (2 * np.pi))

tau = 0.0
k = 1.0
b_norm = True
gr = st.NearestNeighbor(ar, f, k, tau, b_norm)

w = 1.0
wc = 1.0
pll = st.Pll(w, wc)

sr = st.PllSystem(pll, gr)

#m = 2
#tr = st.Twist.get_states(sr, m)
#print tr
#

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
#t = st.Twist.get_states(sys, m)
#print t[0].omega


c = sys.g.get_single_site_coupling(1)
h_sum0 = lambda s: st.Twist.get_coupling_sum(c, n, 0, h, s, k=1)
h_sum1 = lambda s: st.Twist.get_coupling_sum(c, n, 1, h, s, k=1)
s = np.linspace(0, 10, 1000)


tau = np.linspace(0, 1.7, 100)
t1 = tau[75]
f = lambda s: kc * t1 * h_sum0(s) + w * t1 - s
s_min = (w - k) * t1
s_max = (w + k) * t1
s1 = np.linspace(s_min - 2, s_max + 2, 1000)

omega0 = []
tau_omega0 = []
for t in tau:
    tmp = st.Twist.get_omega(w, kc, t, h_sum0, ns=1000)
    for o in tmp:
        omega0.append(o)
        tau_omega0.append(t)


omega1 = []
tau_omega1 = []
for t in tau:
    tmp = st.Twist.get_omega(w, kc, t, h_sum1, ns=1000)
    for o in tmp:
        omega1.append(o)
        tau_omega1.append(t)


plt.figure(5, figsize=mpl_helpers.SIZE_TUW_TRIPLE)
plt.subplot(3, 1, 1)
plt.plot(s, h_sum0(s))

plt.subplot(3, 1, 2)
plt.plot(s1, f(s1))

plt.subplot(3, 1, 3)
plt.plot(tau_omega0, omega0, 'bo', ms=3)
plt.plot(tau_omega1, omega1, 'ro', ms=3)
plt.xlim([0.0, 1.7])
plt.ylim([4.5, 8.0])







print '##########################################################################'
print 'Clean up'
print '##########################################################################'

mpl_helpers.tight_layout_grid_all()
plt.show()
