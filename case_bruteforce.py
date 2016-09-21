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
def simulatePllNetwork(mode,topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, isPlottingTimeSeries=False):
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

	''' PLOT PHASE & FREQUENCY TIME SERIES '''
	if isPlottingTimeSeries:
		out.plotTimeSeries(phi, F, Fc, dt, orderparam, k, delay, F_Omeg, K)

	''' RETURN '''																# return value of mean order parameter, last order parameter, and the variance of r during the last 2T_{\omega}
	return {'mean_order': np.mean(r), 'last_orderP': r[len(r)-1], 'stdev_orderP': np.var(r), 'phases': phi, 'intrinfreq': omega_0, 'coupling_strength': K_0}

def multihelper(phiSr, initPhiPrime0, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiM, domega, diffconstK, plot_Phases_Freq, mode):
	if N > 2:
		phiSr = np.insert(phiSr, 0, initPhiPrime0)								# insert the first variable in the rotated space, constant initPhiPrime0
	phiS = eva.rotate_phases(phiSr, isInverse=False)							# rotate back into physical phase space
	np.random.seed()
	return simulatePllNetwork(mode, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, plot_Phases_Freq)

def multihelper_star(dynparam_fixparam):
	return multihelper(*dynparam_fixparam)

def bruteforceout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, Nsim, phiSr=[], show_plot=True):
	mode = int(2);																# mode=0 -> algorithm usage mode, mode=1 -> single realization mode,
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
	# scan through paramters with given discretization - in rotated phase space, for N oscillators
	print('test & brute force mode with evaluation')

	if Nsim > 0:
		print('Parameters set.')
	else:
		print('Error in parameters - supply: \ncase_[sim_mode].py [topology] [#osci] [K] [F_c] [delay] [F_Omeg] [k] [Tsim] [c] [Nsim] [N entries for the value of the perturbation to oscis]')
		sys.exit(0)

	Tsim 	= Tsim*(1.0/F)				  										# simulation time in multiples of the period of the uncoupled oscillators
	Nsteps 	= int(round(Tsim*Fsim))												# calculate number of iterations -- add output?
	print('total simulation time in multiples of the eigentfrequency:', int(Tsim*F),'\n')

	plot_Phases_Freq = False													# whether or not the phases and frequencies are being plotted
	# choose the value of phi'_0, i.e., where the plane, rectangular to the axis phi'_0 in the rotated phase space, is placed
	# this direction corresponds to the case where all phi_k in the original phase space are equal phi_0==phi_1==...==phi_N-1 or (all) have constant phase differences
	initPhiPrime0 = ( 0.0 * np.pi )
	if N > 2:
		print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)

	twistdelta = ( 2.0 * np.pi * k / float(N) )									# phase difference between neighboring oscillators in a stable m-twist state
	#print('phase differences of',k,'-twist:', twistdelta, '\n')
	if k == 0:
		phiM = np.zeros(N)														# vector mit N entries from 0 increasing by twistdelta for every element, i.e., the initial phase-configuration
	else:																		# in the original phase space
		phiM = np.arange(0.0, N*twistdelta, twistdelta)
		#print('phiM = ', phiM, '\n')
	phiMr = eva.rotate_phases(phiM, isInverse=True)								# calculate phiM in terms of rotated phase space
	# print('phiR =', phiR, '\n')

	if N > 2:
		phiMr[0] = initPhiPrime0												# set first dimension in rotated phase space constant for systems of more than 2 oscillators
		#print('phiMr =', phiMr, '\n')
	# the space about each twist solution is scanned in [phiM-pi, phiM+pi] where phiM are the initial phases of the m-twist under investigation
	if N==2:
		scanValues = np.zeros((N,paramDiscretization), dtype=np.float)			# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
		scanValues[0,:] = np.linspace(phiMr[0]-(np.pi), phiMr[0]+(np.pi), paramDiscretization) # all entries are in rotated, and reduced phase space
		scanValues[1,:] = np.linspace(phiMr[1]-(np.pi), phiMr[1]+(np.pi), paramDiscretization) # all entries are in rotated, and reduced phase space
		#print('row', i,'of matrix with all intervals of the rotated phase space:\n', scanValues[i,:], '\n')

		max_scanValuesRotated_1 = eva.rotate_phases([0., phiMr[1]+np.pi],     isInverse=False) # get the largest length in the direction of the first dimension
		min_scanValuesRotated_1 = eva.rotate_phases([0., phiMr[1]-np.pi],     isInverse=False)
		max_scanValuesRotated_0 = eva.rotate_phases([0., phiMr[0]+2.0*np.pi], isInverse=False) # get the largest length in the direction of the 2nd dimension
		min_scanValuesRotated_0 = eva.rotate_phases([0., 0.],                 isInverse=False)
		delta1 = max_scanValuesRotated_1[0,1] - min_scanValuesRotated_1[0,1]; delta0 = max_scanValuesRotated_0[0,1] - min_scanValuesRotated_0[0,1];

		if ( delta0 > 2.0*np.pi or delta0 < 1.9*np.pi or delta1 > 2.0*np.pi or delta1 < 1.9*np.pi):	 # if the borders of the reduced manifold in rotated phase space exceed the 2pi^N cube, adjust scanValues
			scanValues = np.zeros((N,paramDiscretization), dtype=np.float)		# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
			scanValues[0,:] = np.linspace(phiMr[0], phiMr[0]+delta0, paramDiscretization) # all entries are in rotated, and reduced phase space
			scanValues[1,:] = np.linspace(phiMr[1]-delta1*0.5, phiMr[1]+delta1*0.5, paramDiscretization) # all entries are in rotated, and reduced phase space

		_allPoints = itertools.product(*scanValues)
		allPoints = list(_allPoints)											# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
		allPoints = np.array(allPoints) 										# convert the list to an array
	else:
		# setup a matrix for all N-1 variables but the first, which is set later
		scanValues = np.zeros((N-1,paramDiscretization), dtype=np.float)		# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
		for i in range (0, N-1):												# the different coordinates of the solution, discretize an interval plus/minus pi around each variable
			scanValues[i,:] = np.linspace(phiMr[i+1]-np.pi, phiMr[i+1]+np.pi, paramDiscretization) # all entries are in rotated, and reduced phase space
			#print('row', i,'of matrix with all intervals of the rotated phase space:\n', scanValues[i,:], '\n')

		# max_scanValuesRotated = []; min_scanValuesRotated = [];
		# for i in range (0, N-1):												# the different coordinates of the solution, discretize an interval plus/minus pi around each variable
		# 	print('max_scanValuesRotated: ', max_scanValuesRotated)
		# 	max_scanValuesRotated = eva.rotate_phases(scanValues[i,-1], isInverse=True) # get the largest length in the direction of each dimension
		# 	min_scanValuesRotated = eva.rotate_phases(scanValues[i, 0], isInverse=True) # get the largest length in the direction of each dimension
		# 	delta[i] = max_scanValuesRotated[i] - min_scanValuesRotated[i]
		# 	if ( delta[i] > 2.0*np.pi or delta[i] < 1.9*np.pi ):
		# 		scanValues[i,:] = np.linspace(phiMr[i+1]-delta[i]*0.5, phiMr[i+1]+delta[i]*0.5, paramDiscretization) # all entries are in rotated, and reduced phase space

		_allPoints = itertools.product(*scanValues)
		allPoints = list(_allPoints)											# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
		allPoints = np.array(allPoints) 										# convert the list to an array

	#print( 'all points in rotated phase space:\n', allPoints, '\n type:', type(allPoints), '\n')
	#print(_allPoints, '\n')
	#print(itertools.product(*scanValues))

	t0 = time.time()
	if multiproc:																# multiprocessing option for parameter sweep calulcations
		if N == 2:
			Nsim = paramDiscretization**N
			print('multiprocessing', Nsim, 'realizations')
		else:
			Nsim = paramDiscretization**(N-1)
			print('multiprocessing', paramDiscretization**(N-1), 'realizations')
		pool_data=[];
		freeze_support()
		pool = Pool(processes=numberCores)										# create a Pool object
		pool_data.append( pool.map(multihelper_star, itertools.izip( 			# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
							itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(topology), itertools.repeat(couplingfct), itertools.repeat(F), itertools.repeat(Nsteps),
							itertools.repeat(dt), itertools.repeat(c),itertools.repeat(Fc), itertools.repeat(F_Omeg), itertools.repeat(K), itertools.repeat(N), itertools.repeat(k), itertools.repeat(delay),
							itertools.repeat(phiM), itertools.repeat(domega), itertools.repeat(diffconstK), itertools.repeat(plot_Phases_Freq), itertools.repeat(mode) ) ) )
		results=[]; phi=[]; omega_0=[]; K_0=[]; delays_0=[];
		for i in range(Nsim):
			''' evaluate dictionaries '''
			results.append( [ pool_data[0][i]['mean_order'],  pool_data[0][i]['last_orderP'], pool_data[0][i]['stdev_orderP'] ] )
			phi.append( pool_data[0][i]['phases'] )
			omega_0.append( pool_data[0][i]['intrinfreq'] )
			K_0.append( pool_data[0][i]['coupling_strength'] )
			delays_0.append( pool_data[0][i]['transdelays'] )

		del pool_data; del _allPoints;											# emtpy pool data, allPoints variables to free memory

		print( 'size {omega_0, K_0, results}:', sys.getsizeof(omega_0), '\t', sys.getsizeof(K_0), '\t', sys.getsizeof(results), '\n' )
		omega_0=np.array(omega_0); K_0=np.array(K_0); results=np.array(results);
		# np.savez('results/phases_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), phi=phi) # save phases of trajectories
		del phi; # phi=np.array(phi);
		# delays_0=np.array(delays_0);

		#print( list( pool.map(multihelper_star, itertools.izip( 				# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
		#					itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(topology), itertools.repeat(F), itertools.repeat(Nsteps), itertools.repeat(dt),
		#					itertools.repeat(c),itertools.repeat(Fc), itertools.repeat(F_Omeg), itertools.repeat(K), itertools.repeat(N), itertools.repeat(k), itertools.repeat(delay),
		#					itertools.repeat(phiM), itertools.repeat(plot_Phases_Freq) ) ) ) )
		#print('results:', results)
		print('time needed for execution of simulations in multiproc mode: ', (time.time()-t0), ' seconds')
	else:
		results=[]																# prepare container for results of simulatePllNetwork
		for i in range (allPoints.shape[0]):									# iterate through all points in the N-1 dimensional rotated phase space
			print('calculation #:', i+1, 'of', allPoints.shape[0])
			#print( allPoints[i], '\n')
			#print( 'type of allPoints[i]', type(allPoints[i]))
			#print( 'allPoints[i] =', allPoints[i], '\n')
			phiSr = allPoints[i,:]												# go through all combinations of points in the discretized rotated phase space
			if N > 2:
				phiSr = np.insert(phiSr, 0, initPhiPrime0)						# insert the first variable in the rotated space, constant initPhiPrime0
			print( 'phiSr =', phiSr, '\n')
			phiS = eva.rotate_phases(phiSr, isInverse=False)					# rotate back into physical phase space
			#print( 'type of phiS', type(phiS))
			#print( 'type of initPhiPrime0', type(initPhiPrime0))
			#print( 'phiS = ', phiS, '\n')
			data = simulatePllNetwork(mode, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, plot_Phases_Freq, mode)

			''' evaluate dictionaries '''
			results.append( [ data['mean_order'],  data['last_orderP'], data['stdev_orderP'] ] )
			# phi.append( data['phases'] )
			omega_0.append( data['intrinfreq'] )
			K_0.append( data['coupling_strength'] )
			delays_0.append( data['transdelays'] )
			results = np.array(results)

		del pool_data; del _allPoints;											# emtpy pool data, allPoints variables to free memory

		print( 'size omega_0, K_0, results:', sys.getsizeof(omega_0), '\t', sys.getsizeof(K_0), '\t', sys.getsizeof(results), '\n' )
		omega_0=np.array(omega_0); K_0=np.array(K_0); results=np.array(results);
		del phi; # phi=np.array(phi);
			#print( list( simulatePllNetwork(topology, Fc, F_Omeg, K, N, k, delay, phiS, phiM, plot_Phases_Freq) ) )
		print('results:', results)
		print('time needed for execution of simulations sequentially: ', (time.time()-t0), ' seconds')

	#print( 'the results:\n', results, '\n type:', type(results), '\n')
	#print( 'first value in results:\n', results[0], '\n type:', type(results[0]), '\n')
	#print( np.array(results))

	''' SAVE RESULTS '''
	np.savez('results/orderparam_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), results=results)
	np.savez('results/allInitPerturbPoints_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), allPoints=allPoints)

	''' EXTRA EVALUATION '''
	out.doEvalBruteForce(Fc, F_Omeg, K, N, k, delay, twistdelta, results, allPoints, initPhiPrime0, paramDiscretization, delays_0, show_plot)

	return None

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

	bruteforceout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, Nsim, [], False)
