#!/usr/bin/python

from __future__ import division
from __future__ import print_function
import configparser
from configparser import ConfigParser

import sys
import simulation as sim
import output as out
import evaluation as eva
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
import itertools
from itertools import permutations as permu
from itertools import combinations as combi
import time
import datetime

''' SIMULATION CALL '''
def simulatePllNetwork(mode, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, isPlottingTimeSeries=False):
	''' SIMULATION OF NETWORK '''
	simresult = sim.simulateNetwork(mode,N,F,F_Omeg,K,Fc,delay,dt,c,Nsteps,topology,couplingfct,phiS, phiM, domega, diffconstK)
	phi     = simresult['phases']
	omega_0 = simresult['intrinfreq']
	K_0     = simresult['coupling_strength']
	delays_0= simresult['transdelays']
	# print('type phi:', type(phi), 'phi:', phi)

	''' KURAMOTO ORDER PARAMETER '''
	r = eva.oracle_mTwistOrderParameter(phi[-int(2*1.0/(F*dt)):, :], k)			# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
	orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)					# calculate the m-twist order parameter for all times
	#print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])

	''' RETURN '''																# return value of mean order parameter, last order parameter, and the variance of r during the last 2T_{\omega}
	return {'mean_order':np.mean(r), 'last_orderP':r[len(r)-1], 'stdev_orderP':np.var(r), 'phases': phi, 'intrinfreq': omega_0, 'coupling_strength': K_0, 'transdelays': delays_0, 'orderparameter': orderparam}

def multihelper(phiSr, initPhiPrime0, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiM, domega, diffconstK, plot_Phases_Freq):
	if N > 2:
		phiSr = np.insert(phiSr, 0, initPhiPrime0)								# insert the first variable in the rotated space, constant initPhiPrime0
	phiS = eva.rotate_phases(phiSr, isInverse=False)							# rotate back into physical phase space
	np.random.seed()
	return simulatePllNetwork(mode, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, plot_Phases_Freq)

def multihelper_star(dynparam_fixparam):
	return multihelper(*dynparam_fixparam)

''' MAIN '''
if __name__ == '__main__':
	''' MAIN:

	N		  : integer
				number of oscillators

	topology  : string
		global: fully connected network
		ring  : a 1D-ring (closed boundary conditions)
		chain :	a 1D-chain (open boundary conditions)

		!sqrt(N) needs to be an integer
		square:
		hexagon:
		octagon:
	Fc		   : cut-off frequency of the low-pass LF
	F_Omeg     : frequency of state of initial history
	K          : sensetivity of the PLLs

	k		   : integer 0 <= k <= N-1
				 k-Twist

	delay	   : float

	phiS	   : np.array
				 real-valued 2d matrix or 1d vector of phases
				 in the 2d case the columns of the matrix represent the individual oscillators

	Returns
	-------
	0 		   :  phi0 is not stable for specified k-Twist
	1		   :  phi0 is stable for specified k-Twist	'''

	''' SIMULATION PARAMETER'''

	mode = int(1);																# mode=0 -> algorithm usage mode, mode=1 -> single realization mode,
																				# mode=2 -> brute force scanning mode for parameter interval scans
																				# mode=3 -> calculate many noisy realization for the same parameter set
	params = configparser.ConfigParser()										# initiate configparser object to load parts of the system parameters
	params.read('1params.txt')													# read the 1params.txt file from the python code directory

	multiproc 			= str(params['DEFAULT']['multiproc'])					# for multiprocessing set to True (brute-force mode 2)
	paramDiscretization = int(params['DEFAULT']['paramDiscretization'])			# discretization for brute force scan in rotated phase space with phi'_k
	numberCores 		= int(params['DEFAULT']['numberCores'])					# number of child processes in multiproc mode
	couplingfct 		= str(params['DEFAULT']['couplingfct'])					# coupling function for FokkerPlanckEq mode 3: {sin, cos, triang}
	F 					= float(params['DEFAULT']['F'])							# free-running frequency of the PLLs
	Fsim 				= float(params['DEFAULT']['Fsim'])						# simulate phase model with given sample frequency -- goal: about 100 samples per period
	domega     			= float(params['DEFAULT']['domega'])					# the diffusion constant [variance=2*diffconst] of the gaussian distribution for the intrinsic frequencies
	diffconstK 			= float(params['DEFAULT']['diffconstK'])				# the diffusion constant [variance=2*diffconst] of the gaussian distribution for the coupling strength
	# Tsim 				= int(params['DEFAULT']['Tsim'])						# simulation time in multiples of the period of the uncoupled oscillators

	dt					= 1.0/Fsim												# [ dt = T / #samples ] -> #samples per period... with [ T = 1 / F -> dt = 1 / ( #samples * F ) ]

	now = datetime.datetime.now()												# single realization mode
	plot_Phases_Freq = True										  				# plot phase time series for this realization
	# process arguments -- provided on program call, e.g. python oracle.py [arg0] [arg1] ... [argN]
	topology	= str(sys.argv[1])												# topology: {global, chain, ring, square lattice, hexagonal lattice, osctagon}
	N 		 	= int(sys.argv[2])												# number of oscillators
	K 			= float(sys.argv[3])											# coupling strength
	Fc 			= float(sys.argv[4])											# cut-off frequency of the loop filter
	delay 		= float(sys.argv[5])											# signal transmission delay
	F_Omeg 		= float(sys.argv[6])											# frequency of the synchronized state under investigation - has to be obtained before
	k 			= int(sys.argv[7])												# twist-number, specifies solutions of interest, important for setting initial conditions
	Tsim 		= float(sys.argv[8])											# provide the multiples of the intrinsic frequencies for which the simulations runs
	c 			= float(sys.argv[9])											# provide diffusion constant for GWN process, bzw. sigma^2 = 2*c  --> c = 0.5 variance
	Nsim 		= int(sys.argv[10])												# number of realizations for parameterset -- should be one here
	phiSr 		= np.asarray([float(phi) for phi in sys.argv[11:(11+N)]])		# this input allows to simulate specific points in !rotated phase space plane

	if len(phiSr)==N:
		print('Parameters set, perturbations provided manually in rotated phase space of phases.')
		# choose the value of phi'_0, i.e., where the plane, rectangular to the axis phi'_0 in the rotated phase space, is placed
		# this direction corresponds to the case where all phi_k in the original phase space are equal phi_0==phi_1==...==phi_N-1 or (all) have constant phase differences
		initPhiPrime0 = 0.0
		print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
		phiSr[0] = initPhiPrime0							  					# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
		print('\nvalues of the initial phases in ROTATED phase space, i.e., last time-step of history set as initial condition:', phiSr)
		phiS = eva.rotate_phases(phiSr, isInverse=False)		  				# rotate back into physical phase space for simulation
		print('values of initial phase in ORIGINAL phase space:', phiS, '\n')
	else:
		print('Error in parameters - supply: \ncase_[sim_mode].py [topology] [#osci] [K] [F_c] [delay] [F_Omeg] [k] [Tsim] [c] [Nsim] [N entries for the value of the perturbation to oscis]')
		# sys.exit(0)
		phiSValues = np.zeros((Nsim, N), dtype=np.float)						# create vector that will contain the initial perturbation (in the history) for each realizations
		print('\nNo perturbation set, hence all perturbations have the default value zero (in original phase space of phases)!')

	Tsim 	= 2.0*Tsim*(1.0/F)			  										# simulation time in multiples of the period of the uncoupled oscillators
	print('NOTE: single realizations will be simulated for 2*Tsim to have enough waveforms after transients have decayed to plot spectral density.')
	Nsteps 	= int(round(Tsim*Fsim))												# calculate number of iterations -- add output?
	Nsim 	= 1
	print('test single evaluation and plot phase and frequency time series, PROVIDE initial condition in ROTATED phase space!')
	print('total simulation time in multiples of the eigentfrequency:', int(Tsim*F),'\n')

	twistdelta = ( 2.0 * np.pi * k / ( float( N ) ) )							# phase difference between neighboring oscillators in a stable m-twist state
	# print('phase differences of',k,'-twist:', twistdelta, '\n')
	if k == 0:
		phiM = np.zeros(N)														# phiM denotes the unperturbed initial phases according to the m-twist state under investigation
	else:
		phiM = np.arange(0.0, N*twistdelta, twistdelta)							# vector mit N entries from 0 increasing by twistdelta for every element, i.e., the phase-configuration
																				# in the original phase space of an m-twist solution
	t0 = time.time()
	data = simulatePllNetwork(mode, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, plot_Phases_Freq) # initiates simulation and saves result in results container
	print('time needed for execution of simulation: ', (time.time()-t0), ' seconds')

	''' evaluate dictionaries '''
	results=[]; phi=[]; omega_0=[]; K_0=[]; delays_0=[]; orderparam=[];			# prepare container for results of simulatePllNetwork
	results.append( [ data['mean_order'],  data['last_orderP'], data['stdev_orderP'] ] )
	phi.append( data['phases'] )
	omega_0.append( data['intrinfreq'] )
	K_0.append( data['coupling_strength'] )
	delays_0.append( data['transdelays'] )
	orderparam.append( data['orderparameter'] )

	phi=np.array(phi); omega_0=np.array(omega_0); K_0=np.array(K_0); delays_0=np.array(delays_0);
	results=np.array(results); orderparam=np.array(orderparam);

	''' PLOT PHASE & FREQUENCY TIME SERIES '''
	if plot_Phases_Freq:
		out.plotTimeSeries(phi, F, Fc, dt, orderparam, k, delay, F_Omeg, K, couplingfct, Tsim, Fsim)

	plt.show()
	''' SAVE RESULTS '''
	np.savez('results/orderparam_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), results=results)
	np.savez('results/initialperturb_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), phiS=phiS)
