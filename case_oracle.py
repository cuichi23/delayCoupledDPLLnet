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
from multiprocessing import Pool, freeze_support
import itertools
from itertools import permutations as permu
from itertools import combinations as combi
import time
import datetime

''' SIMULATION CALL '''
def simulatePllNetwork(mode,topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, Nx=0, Ny=0, kx=0, ky=0, isPlottingTimeSeries=False):
	''' SIMULATION OF NETWORK '''
	simresult = sim.simulateNetwork(mode,N,F,F_Omeg,K,Fc,delay,dt,c,Nsteps,topology,couplingfct,phiS,phiM,domega,diffconstK,Nx,Ny)
	phi		  = simresult['phases']
	omega_0   = simresult['intrinfreq']
	K_0       = simresult['coupling_strength']
	delays_0  = simresult['transdelays']
	# print('type phi:', type(phi), 'phi:', phi)

	''' KURAMOTO ORDER PARAMETER '''
	r = eva.oracle_mTwistOrderParameter(phi[-int(2*1.0/(F*dt)):, :], k)			# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
	orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)					# calculate the m-twist order parameter for all times
	#print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])

	''' PLOT PHASE & FREQUENCY TIME SERIES '''
	if isPlottingTimeSeries:
		out.plotTimeSeries(phi, F, Fc, dt, orderparam, k, delay, F_Omeg, K)

	''' RETURN '''																# return value of mean order parameter, last order parameter, and the variance of r during the last 2T_{\omega}
	return {'mean_order': np.mean(r), 'last_orderP': r[len(r)-1], 'stdev_orderP': np.var(r), 'phases': phi,
	 		'intrinfreq': omega_0, 'coupling_strength': K_0, 'transdelays': delays_0}

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

	print('Note: here the cases with 2d-twist solutions still needs to be implemented!')

	mode = int(0);																# mode=0 -> algorithm usage mode, mode=1 -> single realization mode,
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

	now = datetime.datetime.now()
	# print('algorithm mode (Josefine)')
	# process arguments -- provided on program call, e.g. python oracle.py [arg0] [arg1] ... [argN], call, e.g.: oracle.py ring 3 1 0.25 0.1 1.15 1 1. 1. 1. 0.0 0.0 0.0
	topology	= str(sys.argv[1])												# topology: {global, chain, ring, square-open, square-periodic, hexagonal lattice, osctagon}
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

	Tsim 		= Tsim*(1.0/F)  												# simulation time in multiples of the period of the uncoupled oscillators
	Nsteps 		= int(round(Tsim*Fsim))											# calculate number of iterations -- add output?
	Nsim=1;

	# choose the value of phi'_0, i.e., where the plane, rectangular to the axis phi'_0 in the rotated phase space, is placed; this direction corresponds to the case where all phi_k
	# in the original phase space are equal phi_0==phi_1==...==phi_N-1 or (all) have constant phase differences; if unequal from zero, that causes a global phase shift in the history
	initPhiPrime0 = 0.0
	# we rotated the original phase space such that along the (new) phi'_0 axis all phase differences remain equal and keep the associated phase constant,
	# considering only the perpendicular plane spanned by phi'_1, phi'_2, ..., phi'_N; if non-zero however, it introduces a global phase shift into the history, implying a different history
	phiSr[0] = initPhiPrime0
	# if looking for m-twists, this variable holds the phase difference between neighboring oscillators in a stable m-twist state
	if ( topology == 'square-open' or topology == 'square-periodic' ):
		print('NOTE: these topologies need parameters Nx, Ny, mx, my, the numbers of oscis in the 2d square grid in x- and y-direction and the twist number in these directions accordingly.')
		print('Make the changes necessary to case_oracle.py!')
		twistdelta_x = ( 2.0 * np.pi * kx / ( float( Nx ) ) )					# phase difference between neighboring oscillators in a stable m-twist state
		twistdelta_y = ( 2.0 * np.pi * ky / ( float( Ny ) ) )					# phase difference between neighboring oscillators in a stable m-twist state
		# print('phase differences of',k,'-twist:', twistdelta, '\n')
		if (k == 0 and kx == 0 and ky == 0):
			phiM = np.zeros(N)													# phiM denotes the unperturbed initial phases according to the m-twist state under investigation
		else:
			phiM=[]
			for rows in range(Ny):												# set the mx-my-twist state's initial condition (history of "perfect" configuration)
				phiMtemp = np.arange(twistdelta_y*rows, Nx*twistdelta_x+twistdelta_y*rows, twistdelta_x)
				phiM.append(phiMtemp)
			phiM = np.array(phiM)
			phiM = phiM.flatten()
	else:
		twistdelta = ( 2.0 * np.pi * k / ( float( N ) ) )						# phase difference between neighboring oscillators in a stable m-twist state
		# print('phase differences of',k,'-twist:', twistdelta, '\n')
		if (k == 0 and kx == 0 and ky == 0):
			phiM = np.zeros(N)													# phiM denotes the unperturbed initial phases according to the m-twist state under investigation
		else:
			phiM = np.arange(0.0, N*twistdelta, twistdelta)						# vector mit N entries from 0 increasing by twistdelta for every element, i.e., the phase-configuration
																				# in the original phase space of an m-twist solution

	phiS = eva.rotate_phases(phiSr, isInverse=False)							# rotate initial phases into physical phase space of phases for simulation
	# check input values -- we only want to check in a 2pi periodic interval [phiS'-pi, phiS'+pi] (for all phiS) around each solution in phase-space
	# with that we are using the periodicity of m-twist solutions in the phase-space of phases (rotated phase space!)
	if any(phiSr[:]<-np.pi) | any(phiSr[:]> np.pi):
		print(0)
	else:
		# print(simulatePllNetwork(mode, topology, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, False))
		data = simulatePllNetwork(mode, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK,  False)
		results = np.array( [ data['mean_order'],  data['last_orderP'], data['stdev_orderP'] ] )
		phi     = data['phases']
		omega_0 = data['intrinfreq']
		K_0     = data['coupling_strength']

		print(results)
