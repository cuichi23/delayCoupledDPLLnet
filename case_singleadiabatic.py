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
def simulatePllNetwork(mode,div,topology,couplingfct,histtype,F,Nsteps,dt,c,Fc,F_Omeg,K,N,k,delay,feedback_delay,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Trelax,Kadiab_value_r,Nx=0,Ny=0,kx=0,ky=0,isPlottingTimeSeries=False):
	''' SIMULATION OF NETWORK '''
	simresult = sim.simulateNetwork(mode,div,N,F,F_Omeg,K,Fc,delay,feedback_delay,dt,c,Nsteps,topology,couplingfct,histtype,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx,Ny,Trelax,Kadiab_value_r)
	phi      = simresult['phases']
	omega_0  = simresult['intrinfreq']
	K_0      = simresult['coupling_strength']
	delays_0 = simresult['transdelays']
	cPD_t    = simresult['cPD']
	Kadiab_t= simresult['Kadiab_t']
	# print('cPD_t:',cPD_t)
	# print('type phi:', type(phi), 'phi:', phi)

	''' MODIFIED KURAMOTO ORDER PARAMETERS '''
	if F > 0:																	# for f=0, there would otherwies be a float division by zero
		F1=F
	else:
		F1=F+1E-3
	if topology == "square-periodic" or topology == "hexagon-periodic" or topology == "octagon-periodic":
		r = eva.oracle_mTwistOrderParameter2d(phi[-int(2*1.0/(F1*dt)):, :], Nx, Ny, kx, ky)
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
		r = eva.oracle_CheckerboardOrderParameter2d(phi[-int(2*1.0/(F1*dt)):, :], Nx, Ny, ktemp)
		# ry = np.nonzero(rmat > 0.995)[0]
		# rx = np.nonzero(rmat > 0.995)[1]
		orderparam = eva.oracle_CheckerboardOrderParameter2d(phi[:, :], Nx, Ny, ktemp)
	elif topology == "chain":
		"""
				k  > 0 : x  checkerboard state
				k == 0 : in-phase synchronized
			"""
		r = eva.oracle_CheckerboardOrderParameter1d(phi[-int(2*1.0/(F1*dt)):, :], k)
		orderparam = eva.oracle_CheckerboardOrderParameter1d(phi[:, :])			# calculate the order parameter for all times
	elif ( topology == "ring" or topology == 'global'):
		r = eva.oracle_mTwistOrderParameter(phi[-int(2*1.0/(F1*dt)):, :], k)	# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
		orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)				# calculate the m-twist order parameter for all times

	# r = eva.oracle_mTwistOrderParameter(phi[-int(2*1.0/(F1*dt)):, :], k)			# calculate the m-twist order parameter for a time interval of 2 times the eigenperiod, ry is imaginary part
	# orderparam = eva.oracle_mTwistOrderParameter(phi[:, :], k)					# calculate the m-twist order parameter for all times
	# print('mean of modulus of the order parameter, R, over 2T:', np.mean(r), ' last value of R', r[-1])

	results=[];
	results.append( [ np.mean(r),  r[len(r)-1], np.var(r) ] )
	now = datetime.datetime.now()												# single realization mode
	''' cut phases if too long time-series '''
	if Nsteps > int(300*1.0/(F1*dt)):
		phi_end = phi[-int(20*1.0/(F1*dt)):, :]
		phi_sta = phi[1:int(50*1.0/(F1*dt)), :]
		phi_cut = []; phi_cut.append( [phi_sta, phi_end] )
		print('\nNOTE: cut out a piece of the phase time-series, since length exceeded Tmax=%i, only saved Ts-> 1 to %i and Te-> -%i to %i' %(int(300*1.0/(F1*dt)),int(50*1.0/(F1*dt)),int(20*1.0/(F1*dt)),Nsteps))
	else:
		phi_cut = phi;

	''' SAVE RESULTS '''
	np.savez('results/PRE_orderparam_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), results=results)
	np.savez('results/PRE_initialperturb_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), phiS=phiS)
	np.savez('results/PRE_phases_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), phases=phi_cut)
	np.savez('results/PRE_Kadiab_t_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), Kadiab_t=Kadiab_t)
	np.savez('results/PRE_cPD_t_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), cPD_t=cPD_t)

	''' RETURN '''																# return value of mean order parameter, last order parameter, and the variance of r during the last 2T_{\omega}
	return {'mean_order':np.mean(r), 'last_orderP':r[len(r)-1], 'stdev_orderP':np.var(r), 'phases': phi,
			'intrinfreq': omega_0, 'coupling_strength': K_0, 'transdelays': delays_0, 'orderparameter': orderparam, 'cPD': cPD_t, 'Kadiab_t': Kadiab_t}

# def multihelper(phiSr, initPhiPrime0, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiM, domega, diffconstK, plot_Phases_Freq, mode):
# 	if N > 2:
# 		phiSr = np.insert(phiSr, 0, initPhiPrime0)								# insert the first variable in the rotated space, constant initPhiPrime0
# 	phiS = eva.rotate_phases(phiSr, isInverse=False)							# rotate back into physical phase space
# 	np.random.seed()
# 	return simulatePllNetwork(mode,div, topology, couplingfct, F, Nsteps, dt, c, Fc, F_Omeg, K, N, k, delay, phiS, phiM, domega, diffconstK, Nx, Ny, kx, ky, plot_Phases_Freq)
#
# def multihelper_star(dynparam_fixparam):
# 	return multihelper(*dynparam_fixparam)

def singleadiabatic(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, cPD, Trelax, Kadiab_value_r, Nsim, Nx=0, Ny=1, kx=0, ky=0, phiConfig=[], phiSr=[], show_plot=True):

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
	diffconstSendDelay	= float(params['DEFAULT']['diffconstSendDelay'])		# the diffusion constant [variance=2*diffconst] of the gaussian distribution for the transmission delays
	feedback_delay		= float(params['DEFAULT']['feedbackDelay'])				# feedback delay of the nodes
	histtype			= str(params['DEFAULT']['histtype'])	  				# what history is being set? uncoupled PLLs (uncoupled), or PLLs in the synchronized state under investigation (syncstate)
	div					= int(params['DEFAULT']['division'])					# division factor for cross-coupling signals
	# Tsim 				= int(params['DEFAULT']['Tsim'])						# simulation time in multiples of the period of the uncoupled oscillators

	dt					= 1.0/Fsim												# [ dt = T / #samples ] -> #samples per period... with [ T = 1 / F -> dt = 1 / ( #samples * F ) ]

	now = datetime.datetime.now()												# single realization mode
	plot_Phases_Freq = True										  				# plot phase time series for this realization

	if ( topology == 'entrainOne' or topology == 'entrainAll' ):
		print('Provide phase-configuration for these cases in physical coordinates!')
		#phiM  = eva.rotate_phases(phiSr.flatten(), isInverse=False);
		phiM  = phiConfig;
		special_case = 0;
		if special_case == 1:
			phiS  = np.array([2., 2., 2.]);
			phiSr = eva.rotate_phases(phiS.flatten(), isInverse=True)
		else:
			phiS = eva.rotate_phases(phiSr.flatten(), isInverse=False)
			#print('Calculated phiS=',phiS,' from phiSr=',phiSr,'.\n')
		print('For entrainOne and entrainAll assumed initial phase-configuration of entrained synced state (physical coordinates):', phiM,
				' and on top a perturbation of (rotated coordinates):', phiSr, '  and in (original coordinates):', phiS, '\n')
		#phiS  = phiSr;
		#phiSr =	eva.rotate_phases(phiS.flatten(), isInverse=True)		  		# rotate back into rotated phase space for simulation
		#print('For entrainOne and entrainAll assumed initial phase-configuration of entrained synced state (physical coordinates):', phiS, ' and (rotated coordinates):', phiSr, '\n')
	else:
		phiS=[]
		phiSrtemp = np.array(phiSr)
		# print('\n\nlen(phiSr):', len(temp[0,:]), 'type(phiSr)', type(phiSr), '\nphiSr:', phiSr)
		if len(phiSrtemp.shape)==1 and len(phiSrtemp) > 0:
			if len(phiSrtemp)==N:
				print('Parameters set, perturbations provided manually in rotated phase space of phases.')
				# choose the value of phi'_0, i.e., where the plane, rectangular to the axis phi'_0 in the rotated phase space, is placed
				# this direction corresponds to the case where all phi_k in the original phase space are equal phi_0==phi_1==...==phi_N-1 or (all) have constant phase differences
				initPhiPrime0 = 0.0
				print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
				phiSrtemp[0] = initPhiPrime0						  					# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
				phiSr = phiSrtemp; del phiSrtemp
				print('\nvalues of the initial phases in ROTATED phase space, i.e., last time-step of history set as initial condition:', phiSr)
				phiS = eva.rotate_phases(phiSr.flatten(), isInverse=False)		  		# rotate back into physical phase space for simulation
				print('values of initial phase in ORIGINAL phase space:', phiS, '\n')
		elif len(phiSrtemp.shape)==2:
			if len(phiSrtemp[0,:])==N:
				print('Parameters set, perturbations provided manually in rotated phase space of phases.')
				# choose the value of phi'_0, i.e., where the plane, rectangular to the axis phi'_0 in the rotated phase space, is placed
				# this direction corresponds to the case where all phi_k in the original phase space are equal phi_0==phi_1==...==phi_N-1 or (all) have constant phase differences
				initPhiPrime0 = 0.0
				print('shift along the first axis in rotated phase space, equivalent to phase kick of all oscillators before simulation starts: phi`_0=', initPhiPrime0)
				phiSrtemp[:,0] = initPhiPrime0						  					# set value of the first dimension, phi'_0, the axis along which all phase differences are preserved
				phiSr = phiSrtemp; del phiSrtemp
				print('\nvalues of the initial phases in ROTATED phase space, i.e., last time-step of history set as initial condition:', phiSr)
				phiS = eva.rotate_phases(phiSr.flatten(), isInverse=False)		  		# rotate back into physical phase space for simulation
				print('values of initial phase in ORIGINAL phase space:', phiS, '\n')
		else:
			print('phiSr: ', phiSr)
			print('Either no initial perturbations given, or Error in parameters - supply: \ncase_[sim_mode].py [topology] [#osci] [K] [F_c] [delay] [F_Omeg] [k] [Tsim] [c] [Nsim] [N entries for the value of the perturbation to oscis]')
			# sys.exit(0)
			phiSValues = np.zeros(N, dtype=np.float)								# create vector that will contain the initial perturbation (in the history) for each realizations
			print('\nNo perturbation set, hence all perturbations have the default value zero (in original phase space of phases)!')
			phiS=phiSValues

	# if F > 0.0:
	# 	Tsim   = Tsim*(1.0/F)				  									# simulation time in multiples of the period of the uncoupled oscillators
	# 	print('total simulation time in multiples of the eigentfrequency:', int(Tsim*F),'\n')
	# else:
	# 	print('Tsim Not in multiples of T_omega, since F=0')
	# 	Tsim   = Tsim*2.0														# in case F = 0 Hz
	# 	print('total simulation time in Tsim*2.0:', int(Tsim*2.0),'\n')
	# print('NOTE: single realizations will be simulated for 2*Tsim to have enough waveforms after transients have decayed to plot spectral density.')

	print('in case_singleadiabatic.singleout, F_Omeg:', F_Omeg)
	Nsteps 	= int(round(Tsim*Fsim))												# calculate number of iterations -- add output?
	Nsim 	= 1
	print('Test single evaluation and plot phase and frequency time series, PROVIDE initial condition in ROTATED phase space!')
	print('Total simulation time in multiples of the eigentfrequency:', int(Tsim*F),'\n')

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
			# print('N =', N, '    Nx =', Nx, '    Ny =', Ny, '    k =', k, '    kx =', kx, '    ky =', ky)
			if (k == 0 and kx == 0 and ky == 0):
				phiM = np.zeros(N)												# phiM denotes the unperturbed initial phases according to the m-twist state under investigation
				print('Length, type and shape of phiM:', len(phiM), type(phiM), phiM.shape)
			else:
				phiM=[]
				# print('type phiM at initialization', type(phiM))
				# print('Entering loop over Ny to set initial phiM.')
				for rows in range(Ny):											# set the mx-my-twist state's initial condition (history of "perfect" configuration)
					# print('loop #', rows)
					#phiMtemp = np.arange(twistdelta_y*rows, Nx*twistdelta_x+twistdelta_y*rows, twistdelta_x)
					phiMtemp = twistdelta_x * np.arange(Nx) + twistdelta_y * rows
					# print('phiMtemp=', phiMtemp, '    of type ', type(phiMtemp), '    and length ', len(phiMtemp))
					phiM.append(phiMtemp)
					# print('phiM(list)=', phiMt, '    of type ', type(phiMt))

				phiM = np.array(phiM)
				# print('phiM[1,]', phiM[1,])
				# print('phiM(array)=', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)

				phiMreorder=np.zeros(Nx*Ny); counter=0;
				for i in range(Nx):
					for j in range(Ny):
						# print('counter:', counter)
						phiMreorder[counter]=phiM[i][j]; counter=counter+1;
				phiM = phiMreorder%(2.0*np.pi)
				# print('phiMreorderd: ', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)

				# NOPE phiM = np.reshape(phiM, (np.product(phiM.shape),))
				# phiM = phiM.flatten();
				# phiM = phiM[:][:].flatten();
				# print('phiMflattened: ', phiM, '    of type ', type(phiM), '    and shape ', phiM.shape)
				# print('Length, type and shape of phiMflattened that was generated:', len(phiM), type(phiM), phiM.shape)
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
				phiM = np.array(phiM)%(2.0*np.pi)								# bring back into interval [0 2pi]
				# print('phiM: ', phiM)											# in the original phase space of an m-twist solution
	if topology == 'global':
		phiM = np.zeros(N)


	# print('time-step dt=', dt)
	t0 = time.time()
	data = simulatePllNetwork(mode,div,topology,couplingfct,histtype,F,Nsteps,dt,c,Fc,F_Omeg,K,N,k,delay,feedback_delay,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Trelax,Kadiab_value_r,Nx,Ny,kx,ky,plot_Phases_Freq) # initiates simulation and saves result in results container
	print('time needed for execution of simulation: ', (time.time()-t0), ' seconds')

	''' evaluate dictionaries '''
	results=[]; phi=[]; omega_0=[]; K_0=[]; delays_0=[]; orderparam=[];	cPD_t=[]; Kadiab_t=[] # prepare container for results of simulatePllNetwork
	results.append( [ data['mean_order'],  data['last_orderP'], data['stdev_orderP'] ] )
	phi.append( data['phases'] )
	omega_0.append( data['intrinfreq'] )
	K_0.append( data['coupling_strength'] )
	delays_0.append( data['transdelays'] )
	orderparam.append( data['orderparameter'] )
	cPD_t.append( data['cPD'] )
	Kadiab_t.append( data['Kadiab_t'] )

	phi=np.array(phi); omega_0=np.array(omega_0); K_0=np.array(K_0); delays_0=np.array(delays_0);
	results=np.array(results); orderparam=np.array(orderparam);

	# ''' SAVE RESULTS '''
	# np.savez('results/orderparam_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), results=results)
	# np.savez('results/initialperturb_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), phiS=phiS)
	# np.savez('results/phases_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), phases=phi)

	''' PLOT PHASE & FREQUENCY TIME SERIES '''
	if plot_Phases_Freq:
		out.plotTimeSeries(phi, F, Fc, dt, orderparam, k, kx, ky, delay, F_Omeg, K, N, Nx, Ny, phiM, topology, c, cPD, cPD_t, Kadiab_t, Kadiab_value_r, couplingfct, Trelax, Fsim, show_plot)

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
	topology		= str(sys.argv[1])											# topology: {global, chain, ring, square-open, square-periodic, hexagonal lattice, osctagon}
	N 		 		= int(sys.argv[2])											# number of oscillators
	K 				= float(sys.argv[3])										# coupling strength
	Fc 				= float(sys.argv[4])										# cut-off frequency of the loop filter
	delay 			= float(sys.argv[5])										# signal transmission delay
	F_Omeg 			= float(sys.argv[6])										# frequency of the synchronized state under investigation - has to be obtained before
	k 				= int(sys.argv[7])											# twist-number, specifies solutions of interest, important for setting initial conditions
	Tsim 			= float(sys.argv[8])										# provide the multiples of the intrinsic frequencies for which the simulations runs
	c 				= float(sys.argv[9])										# provide diffusion constant for GWN process, bzw. sigma^2 = 2*c  --> c = 0.5 variance
	Nsim 			= int(sys.argv[10])											# number of realizations for parameterset -- should be one here
	Nx				= int(sys.argv[11])											# number of oscillators in x-direction
	Ny				= int(sys.argv[12])											# number of oscillators in y-direction
	mx				= int(sys.argv[13])											# twist number in x-direction
	my				= int(sys.argv[14])											# twist number in y-direction
	cPD				= float(sys.argv[15])										# diff constant of GWN in LF
	Trelax			= float(sys.argv[16])										# number of steps for the system to relax to "equilibrium"
	Kadiab_value_r 	= float(sys.argv[17])										# max value of coupling strength
	phiConfig 		= np.asarray([float(phi1) for phi1 in sys.argv[18:(18+N)]])		# this input allows to simulate specific points in !rotated phase space plane
	phiSr 			= np.asarray([float(phi2) for phi2 in sys.argv[(18+N):(18+2*N)]])# this input allows to simulate specific points in !rotated phase space plane

	singleadiabatic(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, cPD, Trelax, Kadiab_value_r, Nsim, Nx, Ny, mx, my, phiConfig, phiSr, True)
