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
def simulatePllNetwork(mode,topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, diffconstSendDelay, cLF, Nx=0, Ny=0, kx=0, ky=0, isPlottingTimeSeries=False):
	''' SIMULATION OF NETWORK '''
	simresult = sim.simulateNetwork(mode,N,F,F_Omeg,K,Fc,delay,dt,c,Nsteps,topology,couplingfct,phiS,phiM,domega,diffconstK,diffconstSendDelay,cLF,Nx,Ny)
	phi     = simresult['phases']
	omega_0 = simresult['intrinfreq']
	K_0     = simresult['coupling_strength']
	delays_0= simresult['transdelays']
	# print('type phi:', type(phi), 'phi:', phi)

	''' MODIFIED KURAMOTO ORDER PARAMETERS '''
	if F > 0:																	# for f=0, there would otherwies be a float division by zero
		F1=F
	else:
		F1=F+1E-3
	if topology == "square-periodic" or topology == "hexagon-periodic" or topology == "octagon-periodic":
		r = eva.oracle_mTwistOrderParameter2d(phi[-int(2*1.0/(F*dt)):, :], Nx, Ny, kx, ky)
		orderparam = eva.oracle_mTwistOrderParameter2d(phi[:, :], Nx, Ny, kx, ky)
	elif topology == "square-open" or topology == "hexagon" or topology == "octagon":
		if kx==1 and ky==1:
			ktemp=2
		elif kx==1 and ky==0:
			ktemp=0
		elif kx==0 and ky==1:
			ktemp=1
		elif kx==0 and ky==0:
			ktemp=3
		"""
				k == 0 : x  checkerboard state
				k == 1 : y  checkerboard state
				k == 2 : xy checkerboard state
				k == 3 : in-phase synchronized
			"""
		r = eva.oracle_CheckerboardOrderParameter2d(phi[-int(2*1.0/(F*dt)):, :], Nx, Ny, ktemp)
		# ry = np.nonzero(rmat > 0.995)[0]
		# rx = np.nonzero(rmat > 0.995)[1]
		orderparam = eva.oracle_CheckerboardOrderParameter2d(phi[:, :], Nx, Ny, ktemp)
	elif topology == "chain":
		"""
				k  > 0 : x  checkerboard state
				k == 0 : in-phase synchronized
			"""
		r = eva.oracle_CheckerboardOrderParameter1d(phi[-int(2*1.0/(F*dt)):, :], k)
		orderparam = eva.oracle_CheckerboardOrderParameter1d(phi[:, :])			# calculate the order parameter for all times
	elif ( topology == "ring" or topology == 'global'):
		r = eva.oracle_mTwistOrderParameter(phi[-int(2*1.0/(F*dt)):, :], k)		# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
		orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)				# calculate the m-twist order parameter for all times
	# r = eva.oracle_mTwistOrderParameter(phi[-int(2*1.0/(F*dt)):, :], k)			# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
	# orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)					# calculate the m-twist order parameter for all times
	# print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])

	''' PLOT PHASE & FREQUENCY TIME SERIES '''
	if isPlottingTimeSeries:
		out.plotTimeSeries(phi, F, Fc, dt, orderparam, k, delay, F_Omeg, K, c, cLF)

	''' RETURN '''																# return value of mean order parameter, last order parameter, and the variance of r during the last 2T_{\omega}
	return {'mean_order':np.mean(r), 'last_orderP':r[len(r)-1], 'stdev_orderP':np.var(r), 'phases': phi,
			'intrinfreq': omega_0, 'coupling_strength': K_0, 'transdelays': delays_0}

def multihelper(phiSr, initPhiPrime0, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiM, domega, diffconstK, cLF, Nx, Ny, kx, ky, plot_Phases_Freq, mode):
	if N > 2:
		phiSr = np.insert(phiSr, 0, initPhiPrime0)								# insert the first variable in the rotated space, constant initPhiPrime0
	phiS = eva.rotate_phases(phiSr, isInverse=False)							# rotate back into physical phase space
	np.random.seed()
	return simulatePllNetwork(mode, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, diffconstSendDelay, cLF, Nx, Ny, kx, ky, plot_Phases_Freq)

def multihelper_star(dynparam_fixparam):
	return multihelper(*dynparam_fixparam)

def noisyout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, cLF, Nsim, Nx=0, Ny=0, kx=0, ky=0, phiSr=[], show_plot=True):
	mode = int(3);																# mode=0 -> algorithm usage mode, mode=1 -> single realization mode,
																				# mode=2 -> brute force scanning mode for parameter interval scans
																				# mode=3 -> calculate many noisy realization for the same parameter set
	params = configparser.ConfigParser()										# initiate configparser object to load parts of the system parameters
	params.read('1params.txt')													# read the 1params.txt file from the python code directory

	multiproc 			= str(params['DEFAULT']['multiproc'])					# 1 for multiprocessing set to True (brute-force mode 2)
	paramDiscretization = int(params['DEFAULT']['paramDiscretization'])			# 2 discretization for brute force scan in rotated phase space with phi'_k
	numberCores 		= int(params['DEFAULT']['numberCores'])					# 3 number of child processes in multiproc mode
	couplingfct 		= str(params['DEFAULT']['couplingfct'])					# 4 coupling function for FokkerPlanckEq mode 3: {sin, cos, triang}
	F 					= float(params['DEFAULT']['F'])							# 5 free-running frequency of the PLLs
	Fsim 				= float(params['DEFAULT']['Fsim'])						# 6 simulate phase model with given sample frequency -- goal: about 100 samples per period
	domega     			= float(params['DEFAULT']['domega'])					# 7 the diffusion constant [variance=2*diffconst] of the gaussian distribution for the intrinsic frequencies
	diffconstK 			= float(params['DEFAULT']['diffconstK'])				# 8 the diffusion constant [variance=2*diffconst] of the gaussian distribution for the coupling strength
	diffconstSendDelay	= float(params['DEFAULT']['diffconstSendDelay'])		# the diffusion constant [variance=2*diffconst] of the gaussian distribution for the transmission delays
	# Tsim 				= int(params['DEFAULT']['Tsim'])						# simulation time in multiples of the period of the uncoupled oscillators

	dt					= 1.0/Fsim												# [ dt = T / #samples ] -> #samples per period... with [ T = 1 / F -> dt = 1 / ( #samples * F ) ]

	print('\nnoise instant. freq., diffconst c:', c, '\nnoise control signal, diffconst. cLF:', cLF)

	now = datetime.datetime.now()
	print('many noisy realizations mode with evaluation')						# -- ATTENTION TO SCALING OF NOISE WITH RESPECT TO INTRINSIC FREQUENCIES')

	initPhiPrime0 = 0															# here, this is just set to be handed over to the next modules
	if len(phiSr)==N:
		print('Parameters set, perturbations provided manually in rotated phase space of phases.')
		# phiSr[0] = initPhiPrime0												# should not be set to zero in this case, since we do not intend to exclude this perturbation
		phiS 	   = eva.rotate_phases(phiSr, isInverse=False)					# rotate back into physical phase space
		phiSValues = []
		for i in range (Nsim):
			phiSValues.append(phiS)												# create vector that will contain the initial perturbation (in the history) for each realizations
	else:
		print('Either no initial perturbations given, or Error in parameters - supply: \ncase_[sim_mode].py [topology] [#osci] [K] [F_c] [delay] [F_Omeg] [k] [Tsim] [c] [Nsim] [Nx] [Ny] [kx] [ky] [N entries for the value of the perturbation to oscis]')
		# sys.exit(0)
		phiSValues = np.zeros((Nsim, N), dtype=np.float)						# create vector that will contain the initial perturbation (in the history) for each realizations
		print('\nNo perturbation set, hence all perturbations have the default value zero (in original phase space of phases)!')

	if F > 0.0:
		Tsim   = Tsim*(1.0/F)				  									# simulation time in multiples of the period of the uncoupled oscillators
		print('total simulation time in multiples of the eigentfrequency:', int(Tsim*F),'\n')
	else:
		print('Tsim Not in multiples of T_omega, since F=0')
		Tsim   = Tsim*2.0														# in case F = 0 Hz
		print('total simulation time in Tsim*2.0:', int(Tsim*2.0),'\n')

	Nsteps = int(round(Tsim*Fsim))												# calculate number of iterations -- add output?
	plot_Phases_Freq = False													# whether or not the phases and frequencies are being plotted

	twistdelta=0; cheqdelta=0; twistdelta_x=0; twistdelta_y=0;
	if not ( topology == 'ring' or topology == 'chain' ):
		if topology == 'square-open' or topology == 'hexagon' or topology == 'octagon':
			cheqdelta_x = np.pi 												# phase difference between neighboring oscillators in a stable chequerboard state
			cheqdelta_y = np.pi 												# phase difference between neighboring oscillators in a stable chequerboard state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			if (kx == 0 and ky == 0):
				phiM = np.zeros(N)												# phiM denotes the unperturbed initial phases according to the m-twist state under investigation

			elif (kx != 0 and ky != 0):
				phiM=[]
				for rows in range(Ny):											# set the mx-my-twist state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(cheqdelta_y*rows, Nx*cheqdelta_x+cheqdelta_y*rows, cheqdelta_x)
					phiM.append(phiMtemp)
				phiM = np.array(phiM)%(2.0*np.pi)
				phiM = phiM.flatten(); # print('phiM: ', phiM)
			elif (kx == 0 and ky != 0):											# prepare chequerboard only in y-direction
				phiM=[]
				for rows in range(Ny):											# set the chequerboard state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(0.0, (Nx-1)*cheqdelta_x, cheqdelta_x)
					phiM.append(phiMtemp)
				phiM = np.array(phiM)%(2.0*np.pi)
				phiM = phiM.flatten(); # print('phiM: ', phiM)

			elif (kx != 0 and ky == 0):											# prepare chequerboard only in x-direction
				phiM=[]
				for columns in range(Nx):										# set the chequerboard state's initial condition (history of "perfect" configuration)
					phiMtemp = np.arange(0.0, (Ny-1)*cheqdelta_y, cheqdelta_y)
					phiM.append(phiMtemp)
				phiM = np.array(phiM)%(2.0*np.pi)
				phiM = phiM.flatten(); # print('phiM: ', phiM)
		elif (topology == 'hexagon-periodic' or topology == 'octagon-periodic' or topology == 'square-periodic'):
			twistdelta_x = ( 2.0 * np.pi * kx / ( float( Nx ) ) )				# phase difference between neighboring oscillators in a stable m-twist state
			twistdelta_y = ( 2.0 * np.pi * ky / ( float( Ny ) ) )				# phase difference between neighboring oscillators in a stable m-twist state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			if (k == 0 and kx == 0 and ky == 0):
				phiM = np.zeros(N)												# phiM denotes the unperturbed initial phases according to the m-twist state under investigation
			else:
				phiM=[]
				for rows in range(Ny):											# set the mx-my-twist state's initial condition (history of "perfect" configuration)
					#phiMtemp = np.arange(twistdelta_y*rows, Nx*twistdelta_x+twistdelta_y*rows, twistdelta_x)
					phiMtemp = twistdelta_x * np.arange(Nx) + twistdelta_y * rows
					phiM.append(phiMtemp)
				phiM = np.array(phiM)
				for i in range(Nx):
					for j in range(Ny):
						# print('counter:', counter)
						phiMreorder[counter]=phiM[i][j]; counter=counter+1;
				phiM = phiMreorder%(2.0*np.pi)
				# phiM = phiM.flatten(); # print('phiM: ', phiM)
	if ( topology == 'ring' or topology == 'chain' ):
		if topology == 'chain':
			cheqdelta = np.pi													# phase difference between neighboring oscillators in a stable chequerboard state
			print('phase differences of',k,' chequerboard:', cheqdelta, '\n')
			if k == 0:
				phiM = np.zeros(N)												# phiM denotes the unperturbed initial phases according to the chequerboard state under investigation
			else:
				phiM = np.arange(0.0, N*cheqdelta, cheqdelta)					# vector mit N entries from 0 increasing by twistdelta for every element, i.e., the phase-configuration
				# print('phiM: ', phiM)											# in the original phase space of an chequerboard solution
		else:
			twistdelta = ( 2.0 * np.pi * k / ( float( N ) ) )					# phase difference between neighboring oscillators in a stable m-twist state
			# print('phase differences of',k,'-twist:', twistdelta, '\n')
			if k == 0:
				phiM = np.zeros(N)												# phiM denotes the unperturbed initial phases according to the m-twist state under investigation
			else:
				phiM = np.arange(0.0, N*twistdelta, twistdelta)					# vector mit N entries from 0 increasing by twistdelta for every element, i.e., the phase-configuration
				# print('phiM: ', phiM)											# in the original phase space of an m-twist solution
	if topology == 'global':
		phiM = np.zeros(N)														# for all-to-all coupling we assume no twist states with m > 0


	_allPoints = phiSValues
	phiSValues = list(phiSValues)												# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
	allPoints = np.array(phiSValues) 											# convert the list to an array

	t0 = time.time()
	if multiproc:																# multiprocessing option for parameter sweep calulcations
		print('multiprocessing', Nsim, 'realizations')
		pool_data=[]; data=[]
		freeze_support()
		pool = Pool(processes=numberCores)									# create a Pool object
		pool_data.append( pool.map(multihelper_star, itertools.izip( 		# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
							phiSValues, itertools.repeat(initPhiPrime0), itertools.repeat(topology), itertools.repeat(couplingfct), itertools.repeat(F), itertools.repeat(Nsteps), itertools.repeat(dt),
							itertools.repeat(c),itertools.repeat(Fc), itertools.repeat(F_Omeg), itertools.repeat(K), itertools.repeat(N), itertools.repeat(k), itertools.repeat(delay),
							itertools.repeat(phiM), itertools.repeat(domega), itertools.repeat(diffconstK), itertools.repeat(cLF), itertools.repeat(Nx), itertools.repeat(Ny), itertools.repeat(kx), itertools.repeat(ky),
							itertools.repeat(plot_Phases_Freq), itertools.repeat(mode) ) ) )
		# print('pool_data:', pool_data, 'type(pool_data):', type(pool_data) )
		results=[]; phi=[]; omega_0=[]; K_0=[]; delays_0=[]; #cLF_t=[]
		for i in range(Nsim):
			''' evaluate dictionaries '''
			results.append( [ pool_data[0][i]['mean_order'],  pool_data[0][i]['last_orderP'], pool_data[0][i]['stdev_orderP'] ] )
			phi.append( pool_data[0][i]['phases'] )
			omega_0.append( pool_data[0][i]['intrinfreq'] )
			K_0.append( pool_data[0][i]['coupling_strength'] )
			delays_0.append( pool_data[0][i]['transdelays'] )
			# cLF_t.append( pool_data[0][i]['cLF'] )

		del pool_data; del _allPoints;											# emtpy pool data, allPoints variables to free memory

		print( 'size {phi, omega_0, K_0, results}:', sys.getsizeof(phi), '\t', sys.getsizeof(omega_0), '\t', sys.getsizeof(K_0), '\t', sys.getsizeof(results), '\n' )
		omega_0=np.array(omega_0); K_0=np.array(K_0); results=np.array(results); delays_0=np.array(delays_0);
		# np.savez('results/phases_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), phi=phi) # save phases of trajectories
		phi=np.array(phi);

		# print('data[0]["mean_order"]', data[0]['mean_order'])
		#print( list( pool.map(multihelper_star, itertools.izip( 			# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
		#					itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(topology), itertools.repeat(F), itertools.repeat(Nsteps), itertools.repeat(dt),
		#					itertools.repeat(c),itertools.repeat(Fc), itertools.repeat(F_Omeg), itertools.repeat(K), itertools.repeat(N), itertools.repeat(k), itertools.repeat(delay),
		#					itertools.repeat(phiM), itertools.repeat(plot_Phases_Freq) ) ) ) )
		print('time needed for execution of simulations in multiproc mode: ', (time.time()-t0), ' seconds')
		# print('data:', data, 'type(data[0]):', type(data[0]))
	else:
		results=[]; phi=[]; omega_0=[]; K_0=[]; data=[]; delays_0=[]; cLF_t=[]	# prepare container for results of simulatePllNetwork
		for i in range (allPoints.shape[0]):									# iterate through all points in the N-1 dimensional rotated phase space
			print('calculation #:', i+1, 'of', allPoints.shape[0])
			#print( allPoints[i], '\n')
			#print( 'type of allPoints[i]', type(allPoints[i]))
			#print( 'allPoints[i] =', allPoints[i], '\n')
			#print( 'type of phiS', type(phiS))
			#print( 'phiS = ', phiS, '\n')
			data = simulatePllNetwork(mode, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, diffconstSendDelay, cLF, Nx, Ny, kx, ky, plot_Phases_Freq, mode)

			''' evaluate dictionaries '''
			results.append( [ data['mean_order'],  data['last_orderP'], data['stdev_orderP'] ] )
			phi.append( data['phases'] )
			omega_0.append( data['intrinfreq'] )
			K_0.append( data['coupling_strength'] )
			delays_0.append( data['transdelays'] )
			# cLF_t.append( data['cLF_t'] )

		phi=np.array(phi); omega_0=np.array(omega_0); K_0=np.array(K_0); delays_0=np.array(delays_0);
		results=np.array(results);

		del pool_data; del _allPoints;											# emtpy pool data, allPoints variables to free memory
		print('results:', results, 'type(results):', type(results))
		print('time needed for execution of simulations sequentially: ', (time.time()-t0), ' seconds')

	''' KURAMOTO ORDER PARAMETER '''
	r=[]; orderparam=[];
	if F > 0:																	# for f=0, there would otherwies be a float division by zero
		F1=F
	else:
		F1=F+1E-3
	for i in range (phi.shape[0]):
		r.append(eva.oracle_mTwistOrderParameter(phi[i,-int(2*1.0/(F1*dt)):, :], k))	# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod
		orderparam.append(eva.oracle_mTwistOrderParameter(phi[i,:, :], k))			# calculate the m-twist order parameter for all times

	''' EXTRA EVALUATION '''
	out.doEvalManyNoisy(F, Fc, F_Omeg, K, N, k, delay, c, cLF, domega, twistdelta, results, allPoints, dt, orderparam, r, phi, omega_0, K_0,delays_0, show_plot)
	# print(r'frequency of zeroth osci at the beginning and end of the simulation:, $\dot{\phi}_0(t_{start})=%.4f$, $\dot{\phi}_0(t_{end})=%.4f$  [rad/Hz]', ((phi[0][int(round(delay/dt))+2][0]-phi[0][int(round(delay/dt))+1][0])/(dt)), ((phi[0][-4][0]-phi[0][-5][0])/(dt)) )
	# print('last values of the phases:\n', phi[0,-3:,0])

	''' SAVE RESULTS '''
	now = datetime.datetime.now()
	np.savez('results/orderparam_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cLF%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cLF, now.year, now.month, now.day), results=results)
	# np.savez('results/InitPerturb_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, now.year, now.month, now.day), allPoints=allPoints)
	# np.savez('results/phases_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, now.year, now.month, now.day), phases=phi)

	del results; del allPoints; del initPhiPrime0; del K_0;	del phi;			# emtpy data container to free memory
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
	Fc		   : cut-off frequency of the low-pass LF in Hz
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
	Nx			= int(sys.argv[11])												# number of oscillators in x-direction
	Ny			= int(sys.argv[12])												# number of oscillators in y-direction
	mx			= int(sys.argv[13])												# twist number in x-direction
	my			= int(sys.argv[14])												# twist number in y-direction
	cLF			= float(sys.argv[15])											# diff constant of GWN in LF
	phiSr 		= np.asarray([float(phi) for phi in sys.argv[16:(16+N)]])		# this input allows to simulate specific points in !rotated phase space plane

	noisyout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, cLF, Nsim, Nx, Ny, mx, my, phiSr, True)
