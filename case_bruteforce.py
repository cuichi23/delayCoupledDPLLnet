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
def simulatePllNetwork(mode,topology,couplingfct,histtype,F,Nsteps,dt,c,Fc,F_Omeg,K,N,k,delay,feedback_delay,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx=0,Ny=0,kx=0,ky=0,isPlottingTimeSeries=False):
	''' SIMULATION OF NETWORK & UNIT CELL CHECK '''
	simresult = sim.simulateNetwork(mode,N,F,F_Omeg,K,Fc,delay,feedback_delay,dt,c,Nsteps,topology,couplingfct,histtype,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx,Ny)
	phi     = simresult['phases']
	omega_0 = simresult['intrinfreq']
	K_0     = simresult['coupling_strength']
	delays_0= simresult['transdelays']
	# print('type phi:', type(phi), 'phi:', phi)

	''' MODIFIED KURAMOTO ORDER PARAMETERS '''
	numb_av_T = 3;																# number of periods of free-running frequencies to average over
	if F > 0:																	# for f=0, there would otherwies be a float division by zero
		F1=F
	else:
		F1=F+1E-3
	if topology == "square-periodic" or topology == "hexagon-periodic" or topology == "octagon-periodic":
		r = eva.oracle_mTwistOrderParameter2d(phi[-int(numb_av_T*1.0/(F1*dt)):, :], Nx, Ny, kx, ky)
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
		r = eva.oracle_CheckerboardOrderParameter2d(phi[-int(numb_av_T*1.0/(F1*dt)):, :], Nx, Ny, ktemp)
		# ry = np.nonzero(rmat > 0.995)[0]
		# rx = np.nonzero(rmat > 0.995)[1]
		orderparam = eva.oracle_CheckerboardOrderParameter2d(phi[:, :], Nx, Ny, ktemp)
	elif topology == "chain":
		"""
				k  > 0 : x  checkerboard state
				k == 0 : in-phase synchronized
			"""
		r = eva.oracle_CheckerboardOrderParameter1d(phi[-int(numb_av_T*1.0/(F1*dt)):, :], k)
		orderparam = eva.oracle_CheckerboardOrderParameter1d(phi[:, :])			# calculate the order parameter for all times
	elif ( topology == "ring" or topology == 'global'):
		r = eva.oracle_mTwistOrderParameter(phi[-int(numb_av_T*1.0/(F1*dt)):, :], k)		# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
		orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)				# calculate the m-twist order parameter for all times
	elif ( topology == "entrainOne" or topology == "entrainAll" ):
		phi_constant_expected = phiM;
		r = eva.calcKuramotoOrderParEntrainSelfOrgState(phi[-int(numb_av_T*1.0/(F1*dt)):, :], phi_constant_expected);
		orderparam = eva.calcKuramotoOrderParEntrainSelfOrgState(phi[:, :], phi_constant_expected);
		#r = eva.oracle_mTwistOrderParameter(phi[-int(numb_av_T*1.0/(F1*dt)):, :], k)	# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
		#orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)				# calculate the m-twist order parameter for all times

	# r = eva.oracle_mTwistOrderParameter(phi[-int(2*1.0/(F*dt)):, :], k)			# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
	# orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)					# calculate the m-twist order parameter for all times
	# print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])

	# ''' PLOT PHASE & FREQUENCY TIME SERIES '''
	if (isPlottingTimeSeries):
		phi1=[]; phi1.append(simresult['phases']); phi1=np.array(phi1);
		orderparam1=[]; orderparam1.append(orderparam); orderparam1=np.array(orderparam1);
		Fsim1=int(1.0/dt); Tsim1=round(Nsteps/Fsim1)
		print('Plot realization! np.shape(phi)', np.shape(phi1), '   type(phi1)', type(phi1), '   TSim:', Tsim1, '   Fsim:', Fsim1); cPD_t=[]; K_adiab_t=[];
		# def plotTimeSeries(phi, F, Fc, dt, orderparam, k, delay, F_Omeg, K, c, cPD, cPD_t=[], Kadiab_t=[], K_adiab_r=(-1), coupFct='triang', Tsim=53, Fsim=None, show_plot=True):
		out.plotTimeSeries(phi1, F, Fc, dt, orderparam1, k, delay, F_Omeg, K, c, cPD, cPD_t, K_adiab_t, -1, couplingfct, Nsteps*dt, Fsim1)
	#print('initial order parameter: ', r[0], '\n')

	''' RETURN '''																# return value of mean order parameter, last order parameter, and the variance of r during the last 2T_{\omega}
	return {'mean_order': np.mean(r), 'last_orderP': r[len(r)-1], 'stdev_orderP': np.var(r), # 'phases': phi, --> DO NOT RETURN PHASES...
	 		'intrinfreq': omega_0, 'coupling_strength': K_0, 'transdelays': delays_0}

def multihelper(phiSr,initPhiPrime0,topology,couplingfct,histtype,F,Nsteps,dt,c,Fc,F_Omeg,K,N,k,delay,feedback_delay,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx,Ny,kx,ky,plot_Phases_Freq,mode):
	if N > 2:
		phiSr = np.insert(phiSr, 0, initPhiPrime0)								# insert the first variable in the rotated space, constant initPhiPrime0
	phiS = eva.rotate_phases(phiSr, isInverse=False)							# rotate back into physical phase space
	# print('TEST in multihelper, phiS:', phiS, ' and phiSr:', phiSr)
	unit_cell = eva.PhaseDifferenceCell(N)
	# SO anpassen, dass auch gegen verschobene Einheitszelle geprueft werden kann (e.g. if not k==0...)
	# ODER reicht schon:
	# if not unit_cell.is_inside(( phiS ), isRotated=False):   ???
	# if not unit_cell.is_inside((phiS-phiM), isRotated=False):					# and not N == 2:	# +phiM
	if not unit_cell.is_inside((phiS), isRotated=False):						# NOTE this case is for scanValues set only in -pi to pi
		return {'mean_order': -1., 'last_orderP': -1., 'stdev_orderP': np.zeros(1), 'phases': phiM,
		 		'intrinfreq': np.zeros(1), 'coupling_strength': np.zeros(1), 'transdelays': delay}
	else:
		np.random.seed()
		return simulatePllNetwork(mode,topology,couplingfct,histtype,F,Nsteps,dt,c,Fc,F_Omeg,K,N,k,delay,feedback_delay,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx,Ny,kx,ky,plot_Phases_Freq)

def multihelper_star(dynparam_fixparam):
	return multihelper(*dynparam_fixparam)

def bruteforceout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, cPD, Nsim, Nx=0, Ny=0, kx=0, ky=0, phiConfig=[], phiSr=[], show_plot=True):
	# print('\n\nkx:', kx, '\n', type(kx))
	mode = int(2);																# mode=0 -> algorithm usage mode, mode=1 -> single realization mode,
																				# mode=2 -> brute force scanning mode for parameter interval scans
																				# mode=3 -> calculate many noisy realization for the same parameter set
																				# mode=4 -> calculate adiabatically changing parameter over time
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
	diffconstSendDelay	= float(params['DEFAULT']['diffconstSendDelay'])		# the diffusion constant [variance=2*diffconst] of the gaussian distribution for the transmission delays
	feedback_delay		= float(params['DEFAULT']['feedbackDelay'])				# feedback delay of the nodes
	histtype			= str(params['DEFAULT']['histtype'])	  				# what history is being set? uncoupled PLLs (uncoupled), or PLLs in the synchronized state under investigation (syncstate)
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
	print('total simulation time in multiples of the eigenfrequency:', int(Tsim*F),'\n')

	plot_Phases_Freq = False													# whether or not the phases and frequencies are being plotted
	if plot_Phases_Freq == True:
		print('\nPlotting time-series for one realizations avtivated, see line 145 in case_bruteforce.py!')

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
			twistdelta = twistdelta_x											# important for evaluation
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
				phiMreorder=np.zeros(Nx*Ny); counter=0;
				for i in range(Nx):
					for j in range(Ny):
						# print('counter:', counter)
						phiMreorder[counter]=phiM[i][j]; counter=counter+1;
				phiM = phiMreorder%(2.0*np.pi)
				# phiM = phiM.flatten(); # print('phiM: ', phiM)
	if ( topology == 'ring' or topology == 'chain'):
		if topology == 'chain':
			cheqdelta = np.pi													# phase difference between neighboring oscillators in a stable chequerboard state
			twistdelta = cheqdelta												# important for evaluation, cheqdelta is just a temporary variable here
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
				phiM = np.array(phiM)%(2.0*np.pi)								# bring back into interval [0 2pi]
				# print('phiM: ', phiM)											# in the original phase space of an m-twist solution
	if topology == 'global':
		phiM = np.zeros(N)														# for all-to-all coupling we assume no twist states with m > 0

	if ( topology == 'entrainOne' or topology == 'entrainAll' ):				# in this case the phase-configuration is given by the solution of the dynamical equations
		phiM = phiConfig;


	phiMr = eva.rotate_phases(phiM, isInverse=True)								# calculate phiM in terms of rotated phase space
	print('m-twist, entrainOne, entrainAll or chequerboard phases phiM =', phiM, 'in coordinates of rotated system, phiMr =', phiMr, '\n')

	# choose the value of phi'_0, i.e., where the plane, rectangular to the axis phi'_0 in the rotated phase space, is placed
	# this direction corresponds to the case where all phi_k in the original phase space are equal phi_0==phi_1==...==phi_N-1 or (all) have constant phase differences
	if ( topology == 'entrainOne' or topology == 'entrainAll' ):
		initPhiPrime0 = phiMr[0]
	else:
		initPhiPrime0 = ( 0.0 * np.pi )

	if N > 2:
		print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
		# phiSr[0] = initPhiPrime0												# set first dimension in rotated phase space constant for systems of more than 2 oscillators
		#print('phiMr =', phiMr, '\n')
	# the space about each twist solution is scanned in [phiM-pi, phiM+pi] where phiM are the initial phases of the m-twist under investigation

	# print('CHECK!!!! phiS should ONLY be the relative perturbation! BUT then in multihelper be careful when checking the unit cell')

	if N==2:
		scanValues = np.zeros((N,paramDiscretization), dtype=np.float)			# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
		# scanValues[0,:] = np.linspace(phiMr[0]-(np.pi), phiMr[0]+(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space
		# scanValues[1,:] = np.linspace(phiMr[1]-(np.pi), phiMr[1]+(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space
		scanValues[0,:] = np.linspace(-(np.pi), +(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
		scanValues[1,:] = np.linspace(-(np.pi), +(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
		#print('row', i,'of matrix with all intervals of the rotated phase space:\n', scanValues[i,:], '\n')

		_allPoints 			= itertools.product(*scanValues)
		allPoints 			= list(_allPoints)									# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
		allPoints 			= np.array(allPoints)								# convert the list to an array
		# allPoints_unitCell  = []
		# for point in allPoints:
		# 	if unit_cell.is_inside(point, isRotated=True):
		# 		allPoints_unitCell.append(point)
		# allPoints			= np.array(allPoints_unitCell)
	else:
		# setup a matrix for all N variables/dimensions and create a cube around the origin with side lengths 2pi
		scanValues = np.zeros((N-1,paramDiscretization), dtype=np.float)		# create container for all points in the discretized rotated phase space, +/- pi around each dimension (unit area)
		for i in range (0, N-1):												# the different coordinates of the solution, discretize an interval plus/minus pi around each variable
			# scanValues[i,:] = np.linspace(phiMr[i+1]-np.pi, phiMr[i+1]+np.pi, paramDiscretization) # all entries are in rotated, and reduced phase space
			if i==0:															# theta2 (x-axis)
				#scanValues[i,:] = np.linspace(-(np.pi), +(np.pi), paramDiscretization) 	# all entries are in rotated, and reduced phase space NOTE: adjust unit cell accordingly!
				scanValues[i,:] = np.linspace(-1.0*np.pi, 1.0*np.pi, paramDiscretization)
			else:																# theta3 (y-axis)
				#scanValues[i,:] = np.linspace(-(1.35*np.pi), +(1.35*np.pi), paramDiscretization)
				scanValues[i,:] = np.linspace(-1.35*np.pi, 1.35*np.pi, paramDiscretization)

			#print('row', i,'of matrix with all intervals of the rotated phase space:\n', scanValues[i,:], '\n')

		_allPoints 			= itertools.product(*scanValues)
		allPoints 			= list(_allPoints)									# scanValues is a list of lists: create a new list that gives all the possible combinations of items between the lists
		allPoints 			= np.array(allPoints) 								# convert the list to an array
		# allPoints_unitCell  = []												# prepare container for points in the unit cell in rotated phase space
		# for point in allPoints:													# loop over all points and pick out the one that belong to the unit cell in rotated phase space
		# 	if unit_cell.is_inside(np.insert(point, 0, initPhiPrime0), isRotated=True):
		# 		allPoints_unitCell.append(point)
		# allPoints			= np.array(allPoints_unitCell)						# since we operated in full coordinates, we now drop the first dimension (related to global phase shift)

	# print( 'all points in rotated phase space:\n', allPoints, '\n type:', type(allPoints), '\n')
	#print(_allPoints, '\n')
	#print(itertools.product(*scanValues))

	t0 = time.time()
	if multiproc == 'TRUE':														# multiprocessing option for parameter sweep calulcations
		Nsim = allPoints.shape[0]
		print('multiprocessing', Nsim, 'realizations')
		pool_data=[];															# should this be recated to be an np.array?
		freeze_support()
		pool = Pool(processes=numberCores)										# create a Pool object

		#def multihelper(phiSr,initPhiPrime0,topology,couplingfct,histtype,F,Nsteps,dt,c,Fc,F_Omeg,K,N,k,delay,feedback_delay,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx,Ny,kx,ky,plot_Phases_Freq,mode):
		pool_data.append( pool.map(multihelper_star, itertools.izip( 			# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
							itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(topology), itertools.repeat(couplingfct), itertools.repeat(histtype),
							itertools.repeat(F), itertools.repeat(Nsteps), itertools.repeat(dt), itertools.repeat(c),itertools.repeat(Fc), itertools.repeat(F_Omeg), itertools.repeat(K),
							itertools.repeat(N), itertools.repeat(k), itertools.repeat(delay), itertools.repeat(feedback_delay), itertools.repeat(phiM), itertools.repeat(domega),
							itertools.repeat(diffconstK), itertools.repeat(diffconstSendDelay), itertools.repeat(cPD), itertools.repeat(Nx), itertools.repeat(Ny), itertools.repeat(kx),
							itertools.repeat(ky), itertools.repeat(plot_Phases_Freq), itertools.repeat(mode) ) ) )
		results=[]; phi=[]; omega_0=[]; K_0=[]; delays_0=[]; #cPD_t=[]
		for i in range(Nsim):
			''' evaluate dictionaries '''
			results.append( [ pool_data[0][i]['mean_order'],  pool_data[0][i]['last_orderP'], pool_data[0][i]['stdev_orderP'] ] )
			# phi.append( pool_data[0][i]['phases'] )
			omega_0.append( pool_data[0][i]['intrinfreq'] )
			K_0.append( pool_data[0][i]['coupling_strength'] )
			delays_0.append( pool_data[0][i]['transdelays'] )
			# cPD_t.append( pool_data[0][i]['cPD'] )

		''' SAVE RESULTS '''
		np.savez('results/orderparam_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), results=results)
		np.savez('results/allInitPerturbPoints_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), allPoints=allPoints)
		del pool_data; del _allPoints;											# emtpy pool data, allPoints variables to free memory

		print( 'size {omega_0, K_0, delays_0, results}:', sys.getsizeof(omega_0), '\t', sys.getsizeof(K_0), '\t', sys.getsizeof(delays_0), '\t', sys.getsizeof(results), '\n' )
		omega_0=np.array(omega_0); K_0=np.array(K_0); results=np.array(results);
		# np.savez('results/phases_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), phi=phi) # save phases of trajectories
		# phi=np.array(phi);
		# delays_0=np.array(delays_0);

		#print( list( pool.map(multihelper_star, itertools.izip( 				# this makes a map of all parameter combinations that have to be simulated, itertools.repeat() names the constants
		#					itertools.product(*scanValues), itertools.repeat(initPhiPrime0), itertools.repeat(topology), itertools.repeat(F), itertools.repeat(Nsteps), itertools.repeat(dt),
		#					itertools.repeat(c),itertools.repeat(Fc), itertools.repeat(F_Omeg), itertools.repeat(K), itertools.repeat(N), itertools.repeat(k), itertools.repeat(delay), itertools.repeat(feedback_delay),
		#					itertools.repeat(phiM), itertools.repeat(plot_Phases_Freq) ) ) ) )
		#print('results:', results)
		print('time needed for execution of simulations in multiproc mode: ', (time.time()-t0), ' seconds')
	else:
		plot_Phases_Freq = True
		results=[]; phi=[]; omega_0=[]; K_0=[]; delays_0=[]; #cPD_t=[]    		# prepare container for results of simulatePllNetwork
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
			data = simulatePllNetwork(mode,topology,couplingfct,histtype,F,Nsteps,dt,c,Fc,F_Omeg,K,N,k,delay,feedback_delay,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx,Ny,kx,ky,plot_Phases_Freq)

			''' evaluate dictionaries '''
			np.concatenate((results, [ data['mean_order'],  data['last_orderP'], data['stdev_orderP'] ] ))
			# phi.append( data['phases'] )
			omega_0.append( data['intrinfreq'] )
			K_0.append( data['coupling_strength'] )
			delays_0.append( data['transdelays'] )
			results = np.array(results)

		''' SAVE RESULTS '''
		np.savez('results/orderparam_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), results=results)
		np.savez('results/allInitPerturbPoints_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), allPoints=allPoints)

		print( 'size omega_0, K_0, results:', sys.getsizeof(omega_0), '\t', sys.getsizeof(K_0), '\t', sys.getsizeof(results), '\n' )
		omega_0=np.array(omega_0); K_0=np.array(K_0); results=np.array(results);
		del phi; # phi=np.array(phi);
			#print( list( simulatePllNetwork(topology, Fc, F_Omeg, K, N, k, delay, phiS, phiM, Nx, Ny, kx, ky, plot_Phases_Freq) ) )
		print('results:', results)
		print('time needed for execution of simulations sequentially: ', (time.time()-t0), ' seconds')

	#print( 'the results:\n', results, '\n type:', type(results), '\n')
	#print( 'first value in results:\n', results[0], '\n type:', type(results[0]), '\n')
	#print( np.array(results))

	# ''' SAVE RESULTS '''
	# np.savez('results/orderparam_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), results=results)
	# np.savez('results/allInitPerturbPoints_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), allPoints=allPoints)

	''' EXTRA EVALUATION '''
	out.doEvalBruteForce(Fc, F_Omeg, K, N, k, delay, twistdelta, results, allPoints, initPhiPrime0, phiMr, paramDiscretization, delays_0, twistdelta_x, twistdelta_y, topology, phiConfig, show_plot)

	del results; del allPoints; del initPhiPrime0; del K_0;						# emtpy data container to free memory
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
	topology	= str(sys.argv[1])												# topology: {global, chain, ring, square-open, square-periodic, hexagonal lattice, osctagon}
	N 		 	= int(float(sys.argv[2]))										# number of oscillators
	K 			= float(sys.argv[3])											# coupling strength
	Fc 			= float(sys.argv[4])											# cut-off frequency of the loop filter
	delay 		= float(sys.argv[5])											# signal transmission delay
	F_Omeg 		= float(sys.argv[6])											# frequency of the synchronized state under investigation - has to be obtained before
	k 			= int(float(sys.argv[7]))										# twist-number, specifies solutions of interest, important for setting initial conditions
	Tsim 		= float(sys.argv[8])											# provide the multiples of the intrinsic frequencies for which the simulations runs
	c 			= float(sys.argv[9])											# provide diffusion constant for GWN process, bzw. sigma^2 = 2*c  --> c = 0.5 variance
	Nsim 		= int(float(sys.argv[10]))										# number of realizations for parameterset -- should be one here
	Nx			= int(float(sys.argv[11]))										# number of oscillators in x-direction
	Ny			= int(float(sys.argv[12]))										# number of oscillators in y-direction
	mx			= int(float(sys.argv[13]))										# twist number in x-direction
	my			= int(float(sys.argv[14]))										# twist number in y-direction
	cPD			= float(sys.argv[15])											# diff constant of GWN in LF
	phiConfig	= np.asarray([float(phi) for phi in sys.argv[16:19]])			# the phase-configuration of an entrained state with N=3 clocks, 2 mutually coupled, 1 reference

	bruteforceout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, cPD, Nsim, Nx, Ny, mx, my, phiConfig, [], False)
