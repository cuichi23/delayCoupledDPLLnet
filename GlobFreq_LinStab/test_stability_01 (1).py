# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:16:01 2017

@author: platz
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize

import path_helpers
import mpl_helpers
mpl_helpers.clear_all()

import synctools2 as st



print '\n##########################################################################'
print '1. Compute frequency curve for example parameters'
print '##########################################################################'

n = 3
m = 0
h = st.Triangle(1.0 / (2 * np.pi))
kc = np.pi / 1.0
w = 2 * np.pi
wc = 1.0 * w
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


tau = np.linspace(0, 1.7, 25)
t1 = tau[15]
f = lambda s: kc * t1 * h_sum0(s) + w * t1 - s
s_min = (w - kc) * t1
s_max = (w + kc) * t1
s1 = np.linspace(s_min - 2, s_max + 2, 1000)


#omega0 = []
#tau_omega0 = []
#for t in tau:
#    tmp = st.Twist.get_omega(w, kc, t, h_sum0, ns=1000)
#    for o in tmp:
#        omega0.append(o)
#        tau_omega0.append(t)
#
#
#omega1 = []
#tau_omega1 = []
#for t in tau:
#    tmp = st.Twist.get_omega(w, kc, t, h_sum1, ns=1000)
#    for o in tmp:
#        omega1.append(o)
#        tau_omega1.append(t)
#
#
#plt.figure(1, figsize=mpl_helpers.SIZE_TUW_TRIPLE)
#plt.subplot(3, 1, 1)
#plt.plot(s, h_sum0(s))
#
#plt.subplot(3, 1, 2)
#plt.plot(s1, f(s1))
#
#plt.subplot(3, 1, 3)
#plt.plot(tau_omega0, omega0, 'bo', ms=3)
#plt.plot(tau_omega1, omega1, 'ro', ms=3)
#plt.xlim([0.0, 1.7])
#plt.ylim([4.5, 8.0])


print '\n##########################################################################'
print '2. Construct derivative matrix'
print '##########################################################################'

s0 = st.Twist.get_states(sys, 0)[0]
s1 = st.Twist.get_states(sys, 1)[0]

dphi0 = s0.get_dphi_matrix()
dphi1 = s1.get_dphi_matrix()

d0 = s0.get_coupling_derivative_matrix()
d1 = s1.get_coupling_derivative_matrix()


plt.figure(2, figsize=mpl_helpers.SIZE_TUW_HALF)

plt.subplot(2, 2, 1)
plt.imshow(dphi0, aspect='auto')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(dphi1, aspect='auto')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(d0, aspect='auto')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(d1, aspect='auto')
plt.colorbar()


print '\n##########################################################################'
print '3. Compute eigenvalues'
print '##########################################################################'


e00, v00 = np.linalg.eig(d0)
e10, v10 = np.linalg.eig(d1)

e01, v01 = s0.get_eigensystem()
e11, v11 = s1.get_eigensystem()

print e00
print e01
print v00
print v01

print ''

print e10
print e11
print v10
print v11


print '\n##########################################################################'
print '4. Get d_sum coupling derivative constant'
print '##########################################################################'

d_sum0 = s0.get_coupling_derivative_sum()
d_sum1 = s1.get_coupling_derivative_sum()

print d_sum0
print d_sum1


print '\n##########################################################################'
print '5. Compute single stability'
print '##########################################################################'


def func(l, b, kc, d_sum, tau, alpha, beta):
    x = np.zeros(2)
    x[0] = l[0] + b * l[0]**2 - b * l[1]**2 + kc * d_sum - kc * np.exp(-l[0] * tau) * (np.cos(l[1] * tau) * alpha + np.sin(l[1] * tau) * beta)
    x[1] = l[1] + 2 * b * l[0] * l[1] - kc * np.exp(-l[0] * tau) * (np.cos(l[1] * tau) * beta - np.sin(l[1] * tau) * alpha)
    return x


l0 = []
b = s0.sys.pll.b
kc = s0.sys.g.k
d_sum = s0.get_coupling_derivative_sum()
tau = s0.sys.g.tau
e0, v0 = s0.get_eigensystem()
for e in e0:
    alpha = np.real(e)
    beta = np.imag(e)
    func_root = lambda l: func(l, b, kc, d_sum, tau, alpha, beta)

    l_init = np.array([0.001, 0.5001])
    l_full = optimize.root(func_root, l_init, tol=1e-14)
    l_num = l_full.x[0] + 1j * l_full.x[1]
    l0.append(l_num)
l0 = np.array(l0)
l0 = np.max(np.real(l0))



l1 = []
b = s1.sys.pll.b
kc = s1.sys.g.k
d_sum = s1.get_coupling_derivative_sum()
tau = s1.sys.g.tau
e1, v1 = s1.get_eigensystem()
for e in e1:
    alpha = np.real(e)
    beta = np.imag(e)
    func_root = lambda l: func(l, b, kc, d_sum, tau, alpha, beta)

    l_init = np.array([0.001, 0.5001])
    l_full = optimize.root(func_root, l_init, tol=1e-14)
    l_num = l_full.x[0] + 1j * l_full.x[1]
    l1.append(l_num)
l1 = np.array(l1)
l1 = np.max(np.real(l1))


print l0
print s0.get_stability()
print ''
print l1
print s1.get_stability()



print '\n##########################################################################'
print '6. Compute stability curve'
print '##########################################################################'

tau = np.linspace(0, 1.7, 100)
t0 = []
o0 = []
l0 = []

t1 = []
o1 = []
l1 = []

for t in tau:
    print t
    sys.g.tau = t

    s0 = st.Twist.get_states(sys, 0)
    for s in s0:
        t0.append(s.sys.g.tau)
        o0.append(s.omega)
        l0.append(s.get_stability()[0])

    s1 = st.Twist.get_states(sys, 1)
    for s in s1:
        t1.append(s.sys.g.tau)
        o1.append(s.omega)
        l1.append(s.get_stability()[0])

t0 = np.array(t0)
o0 = np.array(o0)
l0 = np.array(l0)

i0_stab = np.nonzero(l0 < 0.0)[0]
t0_stab = t0[i0_stab]
o0_stab = o0[i0_stab]
l0_stab = l0[i0_stab]

i0_instab = np.nonzero(l0 >= 0.0)[0]
t0_instab = t0[i0_instab]
o0_instab = o0[i0_instab]
l0_instab = l0[i0_instab]


t1 = np.array(t1)
o1 = np.array(o1)
l1 = np.array(l1)

i1_stab = np.nonzero(l1 < 0.0)[0]
t1_stab = t1[i1_stab]
o1_stab = o1[i1_stab]
l1_stab = l1[i1_stab]

i1_instab = np.nonzero(l1 >= 0.0)[0]
t1_instab = t1[i1_instab]
o1_instab = o1[i1_instab]
l1_instab = l1[i1_instab]






plt.figure(6, figsize=mpl_helpers.SIZE_TUW_TRIPLE)
plt.subplot(2, 1, 1)
plt.plot(t0_stab, o0_stab, 'bo')
plt.plot(t0_instab, o0_instab, 'bx')
#plt.plot(t1_stab, o1_stab, 'ro')
#plt.plot(t1_instab, o1_instab, 'rx')
plt.xlim([0.0, 1.7])
#plt.ylim([4.5, 8.0])

plt.subplot(2, 1, 2, sharex=plt.gca())
plt.plot(t0_stab, l0_stab, 'bo')
plt.plot(t0_instab, l0_instab, 'bx')
#plt.plot(t1_stab, l1_stab, 'ro')
#plt.plot(t1_instab, l1_instab, 'rx')
#plt.ylim([-5.15, 2.0])



print '\n##########################################################################'
print '7. Example stability'
print '##########################################################################'

sys.g.tau = 0.34
s = st.Twist.get_states(sys, 0)[0]
mu, sigma = s.get_stability()
o = s.omega
funcs = s.get_stability_functions()


mm = np.linspace(-1.75, 0.0, 100)
ss = np.linspace(0.0, 5.0, 100)
ff1 = np.zeros((len(funcs), len(ss), len(mm)))
ff2 = np.zeros((len(funcs), len(ss), len(mm)))
for ifuncs in range(len(funcs)):
    for iss in range(len(ss)):
        for imm in range(len(mm)):
            x = np.array([mm[imm], ss[iss]])
            ff1[ifuncs, iss, imm] = funcs[ifuncs](x)[0]
            ff2[ifuncs, iss, imm] = funcs[ifuncs](x)[1]



plt.figure(7, figsize=mpl_helpers.SIZE_TUW_FULL)

plt.subplot(2, 2, 1)
#plt.imshow(ff1[0], aspect='auto')
cs = plt.contour(ff1[0])
plt.clabel(cs, inline=1, fontsize=10)
#plt.colorbar()

plt.subplot(2, 2, 2)
#plt.imshow(ff2[0], aspect='auto')
cs = plt.contour(ff2[0])
plt.clabel(cs, inline=1, fontsize=10)
#plt.colorbar()

plt.subplot(2, 2, 3)
#plt.imshow(ff1[1], aspect='auto')
cs = plt.contour(ff1[1])
plt.clabel(cs, inline=1, fontsize=10)
#plt.colorbar()

plt.subplot(2, 2, 4)
#plt.imshow(ff2[1], aspect='auto')
cs = plt.contour(ff2[1])
plt.clabel(cs, inline=1, fontsize=10)
#plt.colorbar()


print '##########################################################################'
print 'Clean up'
print '##########################################################################'

mpl_helpers.tight_layout_grid_all()
plt.show()
