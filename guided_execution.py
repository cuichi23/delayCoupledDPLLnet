from __future__ import print_function
import os
import gc
import psutil
from os import listdir
import shutil
import csv
import sys
import errno
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import configparser
from configparser import ConfigParser

import matplotlib
if not os.environ.get('SGE_ROOT') == None:										# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt

import evaluation as eva
import case_noisy as cnois
import case_bruteforce as cbrut
import case_singleout as csing
import case_singleadiabatic as cadiab

sys.path.append(os.path.abspath('./GlobFreq_LinStab'))
# import synctools
import synctools_interface as synctools

def chooseTopology():															# ask user-input for topology
	a_true = True
	while a_true:
		# get user input to know which topology should be analyzed
		topology = str(raw_input('\nPlease specify topology from 1-dim [chain, ring], 2-dim [square-open, square-periodic, hexagon, octagon, hexagon-periodic, octagon-periodic] or mean-field [global] to be analyzed: '))
		if ( topology == 'square-open' or topology == 'square-periodic' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic' or topology == 'chain' or topology == 'ring' or topology == 'global' ):
			break
		else:
			print('Please provide one of these input-strings: [chain, ring, square-open, square-periodic, hexagon, octagon, hexagon-periodic, octagon-periodic, global]!')

	return str(topology)

def chooseNumber():																# ask user-input for number of oscis in the network
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		N = int(raw_input('\nPlease specify the number [integer] of oscillators from [2, 1E5] in the network: '))
		if int(N) > 1:
			break
		else:
			print('Please provide input as an [integer] in [2, 1000]!')

	return int(N)

def get2DosciNumbers():
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		Nx = raw_input('\nPlease specify the number [integer] of oscillators in x-direction from [2, 1E5] in the network: ')
		Ny = raw_input('\nPlease specify the number [integer] of oscillators in y-direction from [2, 1E5] in the network: ')
		if ( int(Nx) > 1 and int(Ny) > 1 ):
			break
		else:
			print('Please provide input as an [integer] in [2, 1000]!')

	return int(Nx), int(Ny)

def chooseK(mean_int_freq):														# ask user-input for coupling strength
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		K = float(raw_input('\nPlease specify as [float] the coupling strength [Hz]: '))
		if np.abs( float(K) ) < 5.0 * mean_int_freq or mean_int_freq == 0:
			break
		else:
			print('Please provide input as an [float] in [0, 5*f_intrinsic]!')

	return float(K)

def chooseFc():																	# ask user-input for cut-off frequency
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		Fc = float(raw_input('\nPlease specify as [float] the cut-off frequency Fc [Hz]: '))
		if Fc > 0:
			break
		else:
			print('Please provide input as an [float] in (0, 100*f_intrinsic] [Hz]!')

	return float(Fc)

def chooseTSim():																# ask user-input for the simulation time
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		TSim = float(raw_input('\nPlease specify as [float] the new simulation time TSim in [s]: '))
		if TSim > 0:
			break
		else:
			print('Please provide input as an [float] in (0, OO] ;) [s]!')

	return float(TSim)

def chooseTransDelay():															# ask user-input for delay
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		delay = raw_input('\nPlease specify as [float] the transmission delay [s]: ')
		if float(delay)>=0:
			break
		else:
			print('Please provide input as an [float] in [0, 5 * T_intrinsic] [s]!')

	return float(delay)

def chooseTwistNumber(N, topology):												# ask user-input for twist number
	if topology == "chain":
		a_true = True
		while a_true:
			print('\nTopology with open boundary conditions has been chosen and no m-twist states exist. Only chequerboard and in-phase.')
			# get user input on number of oscis in the network
			k = raw_input('\nPlease specify whether to investigare chequerboard [integer>0] or in-phase [0] synchronized state: ')  # (NOTE that for non-periodic b.c. ANY m > 0 yields a chequerboard configuration)
			if ( int(k)>=0 ):
				break
			else:
				print('Please provide input as an [integer] in [0, oo]!')
	elif topology == "square-open" or topology == "hexagon" or topology == "octagon":
		a_true = True
		while a_true:
			print('\n2d topology with open boundary conditions has been chosen and no m-twist states exist. Only chequerboard (in x-, in y- and in x- and y-direction) and in-phase.')
			# get user input on number of oscis in the network
			k = raw_input('\nPlease specify whether to investigare chequerboard in x-, in y- in xy-direction [, ,], or the synchronized state [0]: ')  # (NOTE that for non-periodic b.c. ANY m > 0 yields a chequerboard configuration)
			if ( int(k)>=0 ):
				break
			else:
				print('Please provide input as an [integer] in [0, oo]!')
	else:
		a_true = True
		while a_true:
			# get user input on number of oscis in the network
			k = raw_input('\nPlease specify the m-twist number [integer] in [0, ..., %d] [dimless]: ' %(N-1))
			if ( int(k)>=0 ):
				break
			else:
				print('Please provide input as an [integer] in [0, %d]!' %(N-1))

	return int(k)

def get2DTwistNumbers(Nx, Ny, topology):										# ask user-input for twist number
	if topology == "square-open" or topology == "hexagon" or topology == "octagon" or topology == "chain":
		a_true = True
		while a_true:
			print('\nTopology with open boundary conditions has been chosen and no m-twist states exist. Only chequerboard and in-phase.')
			# get user input on number of oscis in the network
			kx = raw_input('\nPlease specify: chequerboard only in x-direction [0], only in y-direction [1], in x- and y-direction [2], or in-phase [3] synchronized state: ')
			ky = kx
			if ( int(kx)>=0 and int(ky)>=0 ):
				break
			else:
				print('Please provide input as an [integer] in [0, oo]!')
	else:
		a_true = True
		while a_true:
			# get user input on number of oscis in the network
			kx = raw_input('\nPlease specify the mx-twist number [integer] in [0, ..., %d] [dimless]: ' %(Nx-1))
			ky = raw_input('\nPlease specify the my-twist number [integer] in [0, ..., %d] [dimless]: ' %(Ny-1))
			if ( int(kx)>=0 and int(ky)>=0 ):
				break
			else:
				print('Please provide input as an [integer] in [0, Nx/Ny]!')

	return int(kx), int(ky)

def chooseDeltaPert(N):															# ask user-input for delta-perturbation
	b_true = True
	while b_true:
		# get user input on delta-perturbation type
		choice_history = raw_input('\nPlease specify type delta-perturbation from {[1] manual, [2] all zero, [3] 1 of N perturbed, or [4] all perturbed according to a distribution}: ')
		if ( type(choice_history) == str and int(choice_history) > 0 and int(choice_history) < 5):
			rot_vs_orig = chooseRotOrig()
			break
		else:
			print('Please provide integer input from [1, 2, 3,...]!')

	return choice_history, rot_vs_orig

def chooseDiffConst():															# ask user-input for diffusion constant GWN dynamic noise process on the VCO
	b_true = True
	while b_true:
		# get user input on dynamic frequency noise -- provide diffusion constant for GWN process -> results in Wiener process for the phases (integrated instantaneous freqs)
		c = raw_input('\nPlease specify the diffusion constant [float, in Hz**2] for the GWN process on the frequencies of the DPLLs from the range [0, 10*K]: ')
		if ( float(c) >= 0.0 ):
			break
		else:
			print('Please provide [float] input from [0, 10*K]!')

	return float(c)

def chooseLF_DiffConst():														# ask user-input for diffusion constant GWN dynamic noise process on the control signal
	b_true = True
	while b_true:
		# get user input on dynamic frequency noise -- provide diffusion constant for GWN process -> results in Wiener process for the phases (integrated instantaneous freqs)
		cLF = raw_input('\nPlease specify the diffusion constant [float, in Hz**2] for the GWN process on CONTROL SIGNAL of the DPLLs from the range [0, 10*K]: ')
		if ( float(cLF) >= 0.0 ):
			break
		else:
			print('Please provide [float] input from [0, 10*K]!')

	return float(cLF)

def chooseNsim():																# ask user-input for number of realizations for the noisy statistics
	b_true = True
	while b_true:
		# get user input on dynamic frequency noise -- provide diffusion constant for GWN process -> results in Wiener process for the phases (integrated instantaneous freqs)
		Nsim = raw_input('\nPlease specify how many realizations [integer] of the noisy dynamics should be simulated and averaged from [1, 1E5]: ')
		if int(Nsim) >= 1:
			break
		else:
			print('Please provide [integer] input from [1, 1E5]!')

	return int(Nsim)

def chooseDistribution():
	b_true = True
	while b_true:
		# get user input on type of distributed perturbations
		distrib  = raw_input('\nPlease specify whether initial delta-perturbations are uniformly [iid], [normal] or [gamma]-distributed: ')
		if (distrib == 'gamma' or distrib == 'normal' or distrib == 'iid' ):
			break
		else:
			print('Please provide as input-string either [iid], [normal] or [gamma] as the distribution from which the delta-perturbations at t = -dt are drawn!')

	if distrib == 'iid':
		c_true = True
		while c_true:
			# get user input on interval of iid dist. perturbations
			min_pert = raw_input('\nPlease provide LOWER bound of interval from which iid dist. perturbations are drawn as [float] from interval [-pi, pi): ')
			max_pert = raw_input('Please provide UPPER bound of interval from which iid dist. perturbations are drawn as [float] from interval [min_pert, pi): ')
			if ( float(min_pert) <= float(max_pert) and float(max_pert) < np.pi and float(min_pert) >= -np.pi ):
				print('Draw pseudo random numbers, iid-distributed for the the delta-perturbations at t=-dt in from the interval: [%0.3f, %0.3f]' %(float(min_pert), float(max_pert)) )
				return distrib, min_pert, max_pert, 0, 0, 0, 0
				break
			else:
				print('Please provide inputs as [floats] from the interval [-pi, pi)!')
	elif distrib == 'normal':
		d_true = True
		while d_true:
			# get user input on shape and scale of the gamma-dist. perturbations
			meanvaluePert = raw_input('\nPlease provide the mean-value of the gaussian distribution as [float] from (-infty, infty): ')
			diffconstPert = raw_input('Please provide the standard-deviation of the gaussian distribution as [float] from (0, infty): ')
			if ( float(diffconstPert) > 0 ):
				print('Draw pseudo random numbers, normal-distributed for the the delta-perturbations at t=-dt with mean and diffusion constant: [%0.2f, %0.4g]' %(float(meanvaluePert), float(diffconstPert)) )
				return distrib, 0, 0, meanvaluePert, diffconstPert, 0, 0
				break
			else:
				print('Please provide inputs as [floats] with a standard-deviation larger than zero!')
	elif distrib == 'gamma':
		e_true = True
		while e_true:
			# get user input on shape and scale of the gamma-dist. perturbations
			shape = raw_input('\nPlease provide the SHAPE parameter of the gamma-dist. as [integer] in [1, infty): ')
			scale = raw_input('Please provide the SCALE parameter of the gamma-dist. as [float] in (0, infty): ')
			if ( int(shape) >= 1 and float(scale) > 0 ):
				print('Draw pseudo random numbers, gamma-distributed for the the delta-perturbations at t=-dt with shape and scale parameter: [%d, %0.4g]' %(int(shape), float(scale)) )
				return distrib, 0, 0, 0, 0, shape, scale
				break
			else:
				print('Please provide inputs as follows, shape-parameter: [integer] and scale-parameter: [float]!')

def chooseDeltaPertDistAndBounds(N, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, isRotatedPhaseSpace):
	if distrib == 'iid':
				iid_delta_pert = np.random.uniform(float(min_pert), float(max_pert), size=N)
				perturbation_vec = iid_delta_pert
				if isRotatedPhaseSpace == False:
					perturbation_vec = eva.rotate_phases(perturbation_vec, isInverse=True)	# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
				print('iid-dist. perturbation vector:\n', perturbation_vec)
	elif distrib == 'normal':
				normal_delta_pert = np.random.normal(float(meanvaluePert), float(diffconstPert), size=N)
				perturbation_vec = normal_delta_pert
				if isRotatedPhaseSpace == False:
					perturbation_vec = eva.rotate_phases(perturbation_vec, isInverse=True)	# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
				print('normal-dist. perturbation vector: ', perturbation_vec)
	elif distrib == 'gamma':
				gamma_delta_pert = np.random.gamma(int(shape), float(scale), size=N)
				perturbation_vec = gamma_delta_pert
				if isRotatedPhaseSpace == False:
					perturbation_vec = eva.rotate_phases(perturbation_vec, isInverse=True)	# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
				print('gamma-dist. perturbation vector: ', perturbation_vec)

	return perturbation_vec

def simulateOnlyLinStableCases(para_mat_new):
	# print('in SimOnlyLinStabCases: para_mat_new', len(para_mat_new), '\n')
	if any(decay_rate > 0 for decay_rate in para_mat_new[:,7]):
		d_true = True
		while d_true:
			# get user input: simulate also linearly unstable solutions? if yes, correct Tsim
			input_sim_unstab = raw_input('\nPlease specify whether also linearly unstable solutions should be simulated, [y]es/[n]o: ')
			if ( input_sim_unstab == 'y' or input_sim_unstab == 'n' ):
				if input_sim_unstab == 'y':
					# print('\nUncorrected for negative simulation times:\n', para_mat_new)
					para_mat_new[:,9] = np.abs( para_mat_new[:,9] )				# correct for negative simulation times
					# print('\nCorrected for negative simulation times:', para_mat_new)
					return para_mat_new
					break														# breaks the while loop that catches invalid input
				elif input_sim_unstab == 'n':
					# print('\nUncorrected for negative simulation times:\n', para_mat_new)
					para_mat_linStab = []
					for i in range (len(para_mat_new[:,7])):
						if para_mat_new[i,7] < 0:								# if perturbations decay...
						 	para_mat_linStab.append(para_mat_new[i,:])

					para_mat_linStab = np.array(para_mat_linStab)
					# print('\nDeleted linearly unstable parameter sets:\n', para_mat_linStab)
					return para_mat_linStab
					break
			else:
				print('Please [y]es/[n]o input!')
	else:
		print('\nAll new paramter sets are linearly stable!')
		return para_mat_new

def chooseCsvSaveOption(param_cases_csv, para_mat, topology, couplingfct, c):
	# this extracts the existing parameter sets from the data csv file
	# search all lines in csv files that are equal to the input here in para-mat
	exist_set=[]; para_mat_new=[];
	for i in range (len(para_mat[:,0])):
		temp = []																# reset temp container for every loop
		# print('csv-container: ', param_cases_csv, '\n\n')
		# print('para_mat: ', para_mat, '\n\n')
		# print('type(para_mat[0,:]): ', type(para_mat[0,:]), '\n\n')
		# print( '\nTEST\ntype(para_mat[i,10]):', type(para_mat[i,10]), '\ntype(param_cases_csv[`mx`]):', type(param_cases_csv['mx']), '\nint(para_mat[i,10]):', int(para_mat[i,10]) , '\nparam_cases_csv[`mx`]:', (param_cases_csv['mx']), '\n\n' )
		# print( '\n\nlen(para_mat[:,0]):', len(para_mat[:,0]), '\n\n' )
		temp = param_cases_csv.loc[(param_cases_csv['delay']==float(para_mat[i,4])) & (param_cases_csv['Fc']==float(para_mat[i,3]))
									& (param_cases_csv['K']==float(para_mat[i,2])) & (param_cases_csv['N']==int(para_mat[i,0]))
									& (param_cases_csv['c']==float(c)) & (param_cases_csv['topology']==str(topology)) & (param_cases_csv['couplingfct']==str(couplingfct))
									& (param_cases_csv['Nx']==int(para_mat[i,10])) & (param_cases_csv['Ny']==int(para_mat[i,11]))
									& (param_cases_csv['mx']==int(para_mat[i,12])) & (param_cases_csv['my']==int(para_mat[i,13]))].sort_values(by='K') #guided_execution.py:270: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
		if len(temp) == 0:														# if temp is empty
			print('\nno existing cases found for set with values (para_mat[',i,'][:]) found! Here the search goal parameter set:', para_mat[i][:])
			para_mat_new.append(para_mat[i,:])
		else:
			exist_set.append( temp )
			print('\nexisting cases found for set with values (para_mat[',i,'][:]):', para_mat[i][:], '\nthese are:', temp)
	if exist_set==[]:
		print('So far these parameter sets do not exist!')
	else:
		print('existing parameter sets:\n', exist_set, 'type: ', type(exist_set), 'length', len(exist_set), '\nexist_set[0][:]', exist_set[0][:])
	# print('new parameter sets:\n', para_mat_new)
	para_mat_tmp = np.array(para_mat_new)
	if not len(para_mat_tmp) == 0:
		para_mat_new = simulateOnlyLinStableCases(para_mat_tmp)    				# this fct. corrects for negative Tsim if user decides to simulate also linearly unstable solutions
		print('These are the linear stable cases:', para_mat_new)

	b_true = True
	while b_true:
		# get user input on whether new parameter sets should be added to an csv file containing already simulated cases
		decision = raw_input('\nPlease decide whether to save new parameter sets that are simulated to csv-database [y]es/[n]o: ')
		if decision == 'y':
			# print('\nreturn para_mat_new, type and shape:', para_mat_new.flatten(), type(para_mat_new), np.shape(para_mat_new), '\nend')
			c_true = True
			while c_true:
				# get user input on whether new parameter sets should be simulated, or JUST added to an csv file
				decision1 = raw_input('\nSimulate new parameter sets? [y]es/[n]o Otherwise they will just be saved to csv-database! ')
				print('New parameter sets will be saved to csv-database! {K, Fc, delay, Fomeg, m ,Tsim, id, ReLambda, EstSimseconds, topology, c, N, couplingfct, Nx, Ny, mx, my}')
				writeCsvFileNewCases(para_mat_new, topology, couplingfct, c)
				if decision1 == 'y':
					return para_mat_new											# returned for simulation
				elif decision1 == 'n':
					para_mat_new = []											# return empty for simulation - no sim...
					return para_mat_new
				else:
					print('Please provide [y]es/[n]o input!')
			break
		elif decision == 'n':
			print('New parameter sets will not be saved to csv-database! Simulate these cases: ', para_mat_new)
			return para_mat_new
		else:
			print('Please provide [y]es/[n]o input!')


def writeCsvFileNewCases(para_mat_new, topology, couplingfct, c):
	# find last line in csv-file, ask whether new cases should be added, add if reqiured in the proper format (include id, etc....)
	lastIDcsv = len(param_cases_csv.sort_values('id'))+3						# have to add 3 in order to account for the header of the csv file
	# print('In write function! Here para_mat_new[0,7]: ', para_mat_new[0,7])
	with open('GlobFreq_LinStab/DPLLParameters.csv', 'a') as f:				# 'a' means append to file! other modes: 'w' write only and replace existing file, 'r' readonly...
		writer = csv.writer(f, delimiter=',') #, dtype={'K':np.float, 'Fc':np.float, 'delay':np.float, 'F_Omeg':np.float, 'k':np.int, 'Tsim':np.int, 'id':np.int, 'ReLambda':np.float, 'SimSeconds':np.float, 'topology':np.str, 'c':np.float, 'N':np.int, 'couplingfct':np.str, 'Nx':np.int, 'Ny':np.int, 'mx':np.int, 'my':np.int})
		# write row K, Fc, delay, F_Omeg, k, Tsim, id, ReLambda, Tsim, topology, c, Nx, Ny, mx, my
		for i in range (len(para_mat_new[:,0])):
			id_line = lastIDcsv+1+i
			temp = [str(float(para_mat_new[i,2])), str(float(para_mat_new[i,3])), str(float(para_mat_new[i,4])), str(float(para_mat_new[i,6])),
						str(float(para_mat_new[i,5])), str(int(round(float(para_mat_new[i,9])))), str(id_line), str(para_mat_new[i,7]),
						str(int((float(para_mat_new[i,9])/25.0))), str(topology), str(float(c)), str(int(float(para_mat_new[i,0]))), str(couplingfct),
						str(para_mat_new[i,10]), str(para_mat_new[i,11]), str(para_mat_new[i,12]), str(para_mat_new[i,13])] #list of strings, set by the parameters
			print('\nWRITEOUT:', temp, '\n')
			writer.writerow(temp)

	return None

def loopUserInputPert(N):														# ask N times for the value of the perturbation of the k-th oscillator
	count = 0
	perturb_user_set = []
	while count < N:
		c_true = True
		while c_true:
			# get user input of the perturbation
			pert = float(raw_input('\nPlease specify pertubation as [float] in [0, 2pi] of oscillator k=%d: ' %(count+1)))
			if ( float(pert)>=0 or float(pert)<(2.0*np.pi) ):
				perturb_user_set.append(float(pert))
				count = count + 1
				break															# breaks the while loop that catches invalid input
			else:
				print('Please provide value as [float] in [0, 2pi]!')

	return perturb_user_set

def chooseRotOrig():
	a_true = True
	while a_true:
		# get user input on whether to input perturbation in rotated or original phase space
		rot_vs_orig = raw_input('\nPlease specify whether perturbation is provided in original [o] or rotated [r] phase space: ')
		if (rot_vs_orig == 'o' or rot_vs_orig == 'r'):
			a_true = False; break
		else:
			print('Please provide one of these input-strings: [o]riginal (small letter \"O\") or [r]otated!')

	return rot_vs_orig

def setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k):
	if rot_vs_orig == 'r':														# perturbation provided in rotated phase space

		if case == '1':															# case in which the user provides all pertubations manually
			perturbation_vec = loopUserInputPert(N)

		elif case == '2':														# case for which no delta-perturbations are added
			print('No delta-perturbations, all set to zero!\n')
			phiS = np.zeros(N)
			perturbation_vec = phiS												# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
			# break
		elif case == '3':														# case in which only one of the oscillators is perturbed
			phiS = np.zeros(N);
			b_true = True
			while b_true:
				# get user input on number of oscis in the network
				number 	   = int(raw_input('\nWhich oscillator id [integer] in [0, %d] should be perturbed: ' %(N-1) ))
				pert_value = float(raw_input('Please provide perturbation as float in (0, 2pi] to be added to oscillator id=%d: ' %number))
				if (type(number) == int and number >= 0  and number < N and pert_value>0 and pert_value<2.0*np.pi ):
					phiS[number] = pert_value
					break
				else:
					print('Please provide id as an [integer] in [0, %d] and the perturbation as a [float] in (0,2pi]!' %(N-1) )
			perturbation_vec = phiS												# no transformation necessary, since the modules expect to get the values in rotated phase space

		elif case == '4':														# case in which all the oscillators are perturbed randomly from either iid dist. in
																				# [min_pert, max_pert] or a gamma-distribution with shape and scale parameter
			perturbation_vec = chooseDeltaPertDistAndBounds(N, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, True)

	elif rot_vs_orig == 'o':													# perturbation provided in original phase space

		if case == '1':															# case in which the user provides all pertubations manually
			phiS = loopUserInputPert(N)
			perturbation_vec = eva.rotate_phases(phiS, isInverse=True)			# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
			# break
		elif case == '2':														# case for which no delta-perturbations are added
			print('No delta-perturbations, all set to zero!\n')
			phiS = np.zeros(N)
			perturbation_vec = eva.rotate_phases(phiS, isInverse=True)			# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout

		elif case == '3':														# case in which only one the the oscillators is perturbed
			phiS = np.zeros(N);
			b_true = True
			while b_true:
				# get user input on number of oscis in the network
				number 	   = int(raw_input('\nWhich oscillator id [integer] in [0, %d] should be perturbed: ' %(N-1) ))
				pert_value = float(raw_input('Please provide perturbation as float in (0, 2pi] to be added to oscillator id=%d: ' %number))
				if (type(number) == int and number >= 0  and number < N and pert_value>0 and pert_value<2.0*np.pi ):
					phiS[number] = pert_value
					break
				else:
					print('Please provide id as an [integer] in [0, %d] and the perturbation as a [float] in (0,2pi]!' %(N-1) )
			perturbation_vec = eva.rotate_phases(phiS, isInverse=True)			# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout

		elif case == '4':														# case in which all the oscillators are perturbed randomly from either iid dist. in
																				# [min_pert, max_pert] or a gamma-distribution with shape and scale parameter
			perturbation_vec = chooseDeltaPertDistAndBounds(N, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, False)		# call with 'False' flag in order to get perturbations returned in rotated phase space as needed

	print( '\norder parameter for system after delta-perturbation: ', eva.oracle_mTwistOrderParameter(perturbation_vec, k), ' and twist number: ', k, '\n' )
	return perturbation_vec

''' SINGLE REALIZATION, ADIABATIC CHANGE OF c_LF AFTER TRANSIENT DECAY TIME '''
def singleAdiabatChange(params):
	x_true = True
	while x_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input('\nPlease specify which parameters should be changed adiabatically {VCO noise inst. freq [c] or LF induced noise on control signal [cLF]} : ')
		if user_input == 'cLF':
			cLF_value = float(raw_input('\nPlease specify initial value of cLF_s in [Hz*Hz], from there it will adiabatically change to zero = '))
			Trelax = float(raw_input('\nPlease specify the transient decay time Trelax, adiabatic change start after that time in [s] = '))

			print('\nWill start with this cLF-value: ', cLF_value, ', and relaxation time Trelax: ', Trelax, '\n\n')

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			# c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			c       = 0
			Nsim    = 1															# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			isRadian=False														# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, delay, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, np.array([cLF_value, cLF_value]), isRadians=isRadian)
			print('\n\nAdjust code Daniel to fit cLF!!!!')

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			if not ( para_mat == [] and len(para_mat) == 0 and cLF_value == [] and len(cLF_value) == 0):
				plot_out = True

				# for i in range (len(para_mat[:,0])):
				print('\nSTART: python case_singleadiabatic.py '+str(topology)+' '+str(int(para_mat[0,0]))+' '+str(float(para_mat[0,2]))+' '+str(float((para_mat[0,3])))+' '
												+str(float(delay))+' '+str(float(para_mat[0,6]))+' '+str(int(para_mat[0,5]))+' '+str(int(round(float(para_mat[0,9]))))+' '
												+str(c)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(float(cLF_value))+' '+str(Trelax)+'  '.join(map(str, pert)))
				print('Tsim: ', para_mat[0,9])
				# os.system('python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
				# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
				cadiab.singleadiabatic(str(topology), int(para_mat[0,0]), float(para_mat[0,2]), float((para_mat[0,3])), float(delay), float(para_mat[0,6]),
								int(para_mat[0,5]), int(round(float(para_mat[0,9]))), float(c), float(cLF_value), float(Trelax), int(1), int(Nx), int(Ny), int(kx), int(ky),
								pert, plot_out)
				gc.collect()
			break

		elif user_input == 'c':
			c_value = float(raw_input('\nPlease specify initial value of c in [Hz*Hz], from there it will adiabatically change to zero = '))
			Trelax = float(raw_input('\nPlease specify the transient decay time Trelax, adiabatic change start after that time in [s] = '))

			print('\nWill start with this c-value: ', c_value, ', and relaxation time Trelax: ', Trelax, '\n\n')

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			# c 		= chooseDiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF     = 0
			Nsim    = 1															# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before

			# if str(params['DEFAULT']['couplingfct']) == 'triang':			    # set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			isRadian=False													# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, delay, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, np.array([c_value, c_value]), isRadians=isRadian)
			print('\n\nAdjust code Daniel to fit c!!!!')

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)			# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)					# correct for negative Tsim = -25 / Re(Lambda)....

			if not ( para_mat == [] and len(para_mat) == 0 and c_value == [] and len(c_value) == 0):
				plot_out = True

				# for i in range (len(para_mat[:,0])):
				print('\nSTART: python case_singleadiabatic.py '+str(topology)+' '+str(int(para_mat[0,0]))+' '+str(float(para_mat[0,2]))+' '+str(float((para_mat[0,3])))+' '
												+str(float(delay))+' '+str(float(para_mat[0,6]))+' '+str(int(para_mat[0,5]))+' '+str(int(round(float(para_mat[0,9]))))+' '
												+str(c_value)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(float(cLF))+' '+str(Trelax)+'  '.join(map(str, pert)))
				print('Tsim: ', para_mat[0,9])
				# os.system('python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
				# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
				cadiab.singleadiabatic(str(topology), int(para_mat[0,0]), float(para_mat[0,2]), float((para_mat[0,3])), float(delay), float(para_mat[0,6]),
								int(para_mat[0,5]), int(round(float(para_mat[0,9]))), float(c_value), float(cLF), float(Trelax), int(1), int(Nx), int(Ny), int(kx), int(ky),
								pert, plot_out)
				gc.collect()
			break

		else:
			print('Please provide input from the following options: [c , cLF]: ')

	return None

''' SINGLE REALIZATION '''
def singleRealization(params):
	x_true = True
	while x_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input('\nPlease specify which parameters dependencies to be analyzed {coupling strength [K] in [Hz], LF cut-off freq [Fc] in [Hz], transmission [delay] in [s], loop filter GWN noise diff. const. [cLF] in [Hz^2]} : ')
		if user_input == 'K':
			user_sweep_start = float(raw_input('\nPlease specify the range in which K (K=0.5*Kvco) should be simulated, start K_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K (K=0.5*Kvco) should be simulated, end K_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dK = '))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)
			print('Sweep these K-values: ', new_K_values, '\n')

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			Nsim    = 1															# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='NONE'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			# print('params:', params)

			# perform a K sweep
			isRadian=False														# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, new_K_values, delay, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=lower_TSim
				if para_mat[i,9]>upper_TSim:
					para_mat[i,9]=upper_TSim

			if not para_mat == [] and not len(para_mat) == 0:

				if len(para_mat[:,0]) == 1:
					print('Estimated time until perturbations have decayed to exp(-25) times the initial perturbations: ', para_mat[0,9])
					if str( raw_input(' Change? [y]es / [n]o: ') ) == 'y':
						para_mat[0,9] = chooseTSim()
						print('\nNew TSim set.')
					else:
						print('Simulation time remains as calculated/approximated by synctools.py')
				else:
					print('\nCould implement here to change TSim also in the case of parameter sweeps. Length para_mat: ', len(para_mat))

				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('\nSTART: python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
							+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
							+str(c)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF)+' '.join(map(str, pert)))
					print('Tsim: ', para_mat[i,9])
					# os.system('python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					#HILFE: def singleout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, cLF, Nsim, Nx=0, Ny=1, kx=0, ky=0, phiSr=[], show_plot=True)
					csing.singleout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(1), int(Nx), int(Ny), int(kx), int(ky),
									pert, plot_out)
			break

		elif user_input == 'Fc':
			user_sweep_start = float(raw_input('\nPlease specify the range in which Fc should be simulated, start Fc_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which Fc should be simulated, end Fc_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dFc = '))
			new_Fc_values	 = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			Nsim    = 1														# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a Fc sweep
			isRadian=False														# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, delay, str(params['DEFAULT']['couplingfct']), new_Fc_values, k, kx, ky,topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=lower_TSim
				if para_mat[i,9]>upper_TSim:
					para_mat[i,9]=upper_TSim

			if not para_mat == [] and not len(para_mat) == 0:

				if len(para_mat[:,0]) == 1:
					print('Estimated time until perturbations have decayed to exp(-25) times the initial perturbations: ', para_mat[0,9])
					if str( raw_input(' Change? [y]es / [n]o: ') ) == 'y':
						para_mat[0,9] = chooseTSim()
						print('\nNew TSim set.')
					else:
						print('Simulation time remains as calculated/approximated by synctools.py')
				else:
					print('\nCould implement here to change TSim also in the case of parameter sweeps. Length para_mat: ', len(para_mat))

				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('\nSTART: python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
													+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
													+str(c)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF)+' '.join(map(str, pert)))
					print('Tsim: ', para_mat[i,9])
					# os.system('python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					csing.singleout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(1), int(Nx), int(Ny), int(kx), int(ky),
									pert, plot_out)
			break

		elif user_input == 'delay':
			user_sweep_start = float(raw_input('\nPlease specify the range in which delays should be simulated, start delay_s in [s] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which delays should be simulated, end delay_e in [s] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [s] ddelay = '))
			new_delay_values = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			Nsim    = 1														# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			isRadian=False														# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, new_delay_values, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=lower_TSim
				if para_mat[i,9]>upper_TSim:
					para_mat[i,9]=upper_TSim

			if not para_mat == [] and not len(para_mat) == 0:

				if len(para_mat[:,0]) == 1:
					print('Estimated time until perturbations have decayed to exp(-25) times the initial perturbations: ', para_mat[0,9])
					if str( raw_input(' Change? [y]es / [n]o: ') ) == 'y':
						para_mat[0,9] = chooseTSim()
						print('\nNew TSim set.')
					else:
						print('Simulation time remains as calculated/approximated by synctools.py')
				else:
					print('\nCould implement here to change TSim also in the case of parameter sweeps. Length para_mat: ', len(para_mat))

				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('\nSTART: python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
													+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
													+str(c)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF)+' '.join(map(str, pert)))
					print('Tsim: ', para_mat[i,9])
					# os.system('python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					csing.singleout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(1), int(Nx), int(Ny), int(kx), int(ky),
									pert, plot_out)
			break

		elif user_input == 'cLF':
			user_sweep_start = float(raw_input('\nPlease specify the range in which cLF should be simulated, start cLF_s in [Hz*Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which cLF should be simulated, end cLF_e in [Hz*Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz*Hz] dc = '))
			new_cLF_values = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)
			print('\nWill scan these cLF-values: ', new_cLF_values, '\n\n')

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			Nsim    = 1															# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			isRadian=False														# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, delay, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, new_cLF_values, isRadians=isRadian)
			print('\n\nAdjust code Daniel to fit cLF!!!!')

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=lower_TSim
				if para_mat[i,9]>upper_TSim:
					para_mat[i,9]=upper_TSim


			if not ( para_mat == [] and len(para_mat) == 0 and new_cLF_values == [] ):

				if len(para_mat[:,0]) == 1:
					print('Estimated time until perturbations have decayed to exp(-25) times the initial perturbations: ', para_mat[0,9])
					if str( raw_input(' Change? [y]es / [n]o: ') ) == 'y':
						para_mat[0,9] = chooseTSim()
						print('\nNew TSim set.')
					else:
						print('Simulation time remains as calculated/approximated by synctools.py')
				else:
					print('\nCould implement here to change TSim also in the case of parameter sweeps. Length para_mat: ', len(para_mat))

				# print( 'length of para_mat[:,0]:', len(para_mat[:,0]) )
				# print( 'length of new_c_values :', len(new_c_values)  )
				if ( len(para_mat[:,0]) > 1 or len(new_cLF_values) > 1 ):
					plot_out = False
				elif ( len(para_mat[:,0]) == 1 and new_cLF_values == 1 ):
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					for j in range (len(new_cLF_values)):
						print('\nSTART: python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
														+str(float(delay))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
														+str(c)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(float(new_cLF_values[j]))+' '.join(map(str, pert)))
						print('Tsim: ', para_mat[i,9])
						# os.system('python case_singleout.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
						# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
						csing.singleout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float(delay), float(para_mat[i,6]),
										int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(new_cLF_values[j]), int(1), int(Nx), int(Ny), int(kx), int(ky),
										pert, plot_out)
				gc.collect()
			break

		else:
			print('Please provide input from the following options: [K, Fc , delay, cLF]: ')

	return None

''' STATISTICS '''
def noisyStatistics(params):
	x_true = True
	while x_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input('\nPlease specify which parameters dependencies to be analyzed {coupling strength [K] in [Hz], LF cut-off freq [Fc] in [Hz], transmission [delay] in [s], loop filter GWN noise diff. const. cLF in [Hz^2], or [c] the diffusion constant (sigma^2=2*c).} : ')
		if user_input == 'K':
			user_sweep_start = float(raw_input('\nPlease specify the range in which K (K=0.5*Kvco) should be simulated, start K_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K (K=0.5*Kvco) should be simulated, end K_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dK = '))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a K sweep
			isRadian=False														# set this False if values provided in [Hz] instead of [rad * Hz]
			print('K before:', new_K_values)
			sf = synctools.SweepFactory(N, Ny, Nx, F, new_K_values, delay, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)
			# print('{tau, K, w, wc, m, Omega, Re(lambda)}', delay, new_K_values, 2*np.pi*F, 2*np.pi*Fc, k, para_mat[:,6], para_mat[:,7] )
			# print('{tau, K, F, Fc, m, F_Omeg, Re(lambda)}', float(para_mat[0,4]), para_mat[0,2], float(para_mat[0,1]), float(para_mat[0,3]), int(para_mat[0,5]), para_mat[:,6], para_mat[:,7] )

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=80

			if not para_mat == [] and not len(para_mat) == 0:
				print('length of para_mat[:,0]:', len(para_mat[:,0]))
				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('\nSTART: python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
												+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
												+str(c)+' '+str(Nsim)+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF))
					print('Tsim: ', para_mat[i,9])
					# os.system('python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+str(Nsim)ky.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					# def noisyout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, Nsim, Nx=0, Ny=0, kx=0, ky=0, phiSr=[], show_plot=True):
					cnois.noisyout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(Nsim), int(Nx), int(Ny), int(kx), int(ky),
									pert[i], plot_out)
			break

		elif user_input == 'c':
			new_c_values	 = []
			user_sweep_start = float(raw_input('\nPlease specify the range in which c should be simulated, start c_s in [Hz*Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which c should be simulated, end c_e in [Hz*Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz*Hz] dc = '))
			new_c_values	 = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)
			print('\nWill scan these c-values: ', new_c_values, '\n\n')

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before


			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a K sweep
			isRadian=False														# set this False if values provided in [Hz] instead of [rad * Hz]
			print('c before:', new_c_values)
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, delay, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, new_c_values, isRadians=isRadian) # just the determinisic parameters

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=80

			# print('sum pert: ', np.sum(pert))
			# if np.sum(pert) == 0.0:
			# 	print('limit Tsim since there is no perturbation and the system is set up in the synced state')
			# 	para_mat[i,9]=lower_TSim

			if not ( para_mat == [] and new_c_values == [] ):
				# print( 'length of para_mat[:,0]:', len(para_mat[:,0]) )
				# print( 'length of new_c_values :', len(new_c_values)  )
				if ( len(para_mat[:,0]) > 1 or len(new_c_values) > 1 ):
					plot_out = False
				elif ( len(para_mat[:,0]) == 1 and new_c_values == 1 ):
					plot_out = True
				else:
					plot_out = False

				# print('length para_mat[:,6], para_mat[:,6] (Omegas for the value of tau):', len(para_mat[:,6]), para_mat[:,6])
				for i in range (len(para_mat[:,0])):
					for j in range (len(new_c_values)):
						print('\nSTART: python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
												+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
												+str(new_c_values[j])+' '+str(Nsim)+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF))
						# os.system('python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+str(Nsim)ky.join(map(str, pert)))
						# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
						# def noisyout(topology, N, K, Fc, delay, F_Omeg, k, Tsim, c, Nsim, Nx=0, Ny=0, kx=0, ky=0, phiSr=[], show_plot=True):
						cnois.noisyout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
										int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(new_c_values[j]), int(Nsim), int(Nx), int(Ny), int(kx), int(ky),
										pert[i], plot_out)
				gc.collect()
			break

		elif user_input == 'Fc':
			user_sweep_start = float(raw_input('\nPlease specify the range in which Fc should be simulated, start Fc_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which Fc should be simulated, end Fc_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dFc = '))
			new_Fc_values	 = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before


			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a Fc sweep
			isRadian=False														# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, delay, str(params['DEFAULT']['couplingfct']), new_Fc_values, k, kx, ky,topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=lower_TSim
				if para_mat[i,9]>upper_TSim:
					para_mat[i,9]=upper_TSim

			if not para_mat == [] and not len(para_mat) == 0:
				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('\nSTART: python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
												+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
												+str(c)+' '+str(cLF)+' '+str(Nsim)+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+str(cLF)+' '.join(map(str, pert)))
					# os.system('python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+str(Nsim)+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					cnois.noisyout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(Nsim), int(Nx), int(Ny), int(kx), int(ky),
									pert[i], plot_out)
					gc.collect()
			break

		elif user_input == 'cLF':
			user_sweep_start = float(raw_input('\nPlease specify the range in which cLF should be simulated, start cLF_s in [Hz*Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which cLF should be simulated, end cLF_e in [Hz*Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz*Hz] dc = '))
			new_cLF_values = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)
			print('\nWill scan these cLF-values: ', new_cLF_values, '\n\n')

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before


			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a Fc sweep
			isRadian=False														# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, delay, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, new_cLF_values, isRadians=isRadian)
			print('\n\nAdjust code Daniel to fit cLF!!!!')

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=lower_TSim
				if para_mat[i,9]>upper_TSim:
					para_mat[i,9]=upper_TSim

			if not ( para_mat == [] and new_cLF_values == [] ):
				# print( 'length of para_mat[:,0]:', len(para_mat[:,0]) )
				# print( 'length of new_c_values :', len(new_c_values)  )
				if ( len(para_mat[:,0]) > 1 or len(new_cLF_values) > 1 ):
					plot_out = False
				elif ( len(para_mat[:,0]) == 1 and new_cLF_values == 1 ):
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					for j in range (len(new_cLF_values)):
						print('\nSTART: python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float(Fc))+' '
													+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
													+str(c)+' '+str(Nsim)+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+str(new_cLF_values[j])+' '.join(map(str, pert)))
						# os.system('python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+str(Nsim)+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
						# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
						cnois.noisyout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float(Fc), float((para_mat[i,4])), float(para_mat[i,6]),
										int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(new_cLF_values[j]), int(Nsim), int(Nx), int(Ny), int(kx), int(ky),
										pert[i], plot_out)
				gc.collect()
			break

		elif user_input == 'delay':
			user_sweep_start = float(raw_input('\nPlease specify the range in which delays should be simulated, start delay_s in [s] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which delays should be simulated, end delay_e in [s] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [s] ddelay = '))
			new_delay_values = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			case, rot_vs_orig = chooseDeltaPert(N)								# calls function that asks user for input for delta-like perturbation
			if case == '4':
				distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale = chooseDistribution()
			else:
				distrib='gamma'; min_pert=0; max_pert=0; meanvaluePert=0; diffconstPert=0; shape=0; scale=0;
			pert = []
			for i in range (Nsim):
				pert.append(setDeltaPertubation(N, case, rot_vs_orig, distrib, min_pert, max_pert, meanvaluePert, diffconstPert, shape, scale, k))	# calls function that calls delta-like perturbation as choosen before

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			isRadian=False														# set this False to get values returned in [Hz] instead of [rad * Hz]
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, new_delay_values, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat)

			para_mat = simulateOnlyLinStableCases(para_mat)						# correct for negative Tsim = -25 / Re(Lambda)....

			upper_TSim = 2000
			lower_TSim = 50
			for i in range (len(para_mat[:,0])):
				if para_mat[i,9]<10:
					para_mat[i,9]=lower_TSim
				if para_mat[i,9]>upper_TSim:
					para_mat[i,9]=upper_TSim

			if not para_mat == [] and not len(para_mat) == 0:
				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('\nSTART: python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
												+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '
												+str(c)+' '+str(Nsim)+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF)+' '
												.join(map(str, pert)))
					# os.system('python case_noisy.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+str(Nsim)+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					cnois.noisyout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(Nsim), int(Nx), int(Ny), int(kx), int(ky),
									pert[i], plot_out)
			break

		else:
			print('Please provide input from the following options: K, Fc , delay in [s]')

	return None

''' BRUTE FORCE '''
def bruteForce(params, param_cases_csv):
	x_true = True
	while x_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input('\nPlease specify which parameters dependencies to be analyzed {[K] in [Hz], [Fc] in [Hz], [delay] in [s]} : ')
		if user_input == 'K':
			user_sweep_start = float(raw_input('\nPlease specify the range in which K (K=0.5*Kvco) should be simulated, start K_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K (K=0.5*Kvco) should be simulated, end K_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dK = '))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)
			print('new K-values: ', new_K_values)

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
				#if topology == "square-open" and ( kx != 0 or ky != 0 ):
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			# pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			pert = [];
			# Nsim    = chooseNsim()											# calls function that asks user for input for number of realizations

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a K sweep
			isRadian=False
			sf = synctools.SweepFactory(N, Ny, Nx, F, new_K_values, delay, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky, topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat_temp = fsl.get_parameter_matrix(isRadians=False)			# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat_temp, type(para_mat_temp))

			para_mat = chooseCsvSaveOption(param_cases_csv, para_mat_temp, topology, str(params['DEFAULT']['couplingfct']), c) # Nx, Ny, kx, ky contained in para_mat_temp
			# print('Filtered (csv-file) parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat, type(para_mat), len(para_mat))

			if not para_mat == [] and not len(para_mat) == 0:
				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('Tsim: ',     para_mat[i,9])
					print('\nSTART: python case_bruteforce.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
													+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '
													+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF)+' '
													.join(map(str, pert)))
					# os.system('python case_bruteforce.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					# cbrut.bruteforceout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
					# 				int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(1), str(int(Nx)), str(int(Ny)), str(int(kx)), str(int(ky)),
					# 				pert, plot_out)
					cbrut.bruteforceout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(1), int(Nx), int(Ny), int(kx), int(ky),
									pert, plot_out)
			break

		elif user_input == 'Fc':
			user_sweep_start = float(raw_input('\nPlease specify the range in which Fc should be simulated, start Fc_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which Fc should be simulated, end Fc_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dFc = '))
			new_Fc_values	 = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)
			print('new Fc-values: ', new_Fc_values)

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			# pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			pert = []
			# Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a Fc sweep
			isRadian=False
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, delay, str(params['DEFAULT']['couplingfct']), new_Fc_values, k, kx, ky,topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat_temp = fsl.get_parameter_matrix(isRadians=False)			# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat_temp)

			para_mat = chooseCsvSaveOption(param_cases_csv, para_mat_temp, topology, str(params['DEFAULT']['couplingfct']), c)

			if not para_mat == [] and not len(para_mat) == 0:
				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('Tsim: ', para_mat[i,9])
					print('\nSTART: python case_bruteforce.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
													+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '
													+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF)+' '
													.join(map(str, pert)))
					# os.system('python case_bruteforce.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					cbrut.bruteforceout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(1), int(Nx), int(Ny), int(kx), int(ky),
									pert, plot_out)
			break

		elif user_input == 'delay':
			user_sweep_start = float(raw_input('\nPlease specify the range in which delays should be simulated, start delay_s in [s] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which delays should be simulated, end delay_e in [s] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [s] ddelay = '))
			new_delay_values = np.arange(user_sweep_start, user_sweep_end * 1.0001, user_sweep_discr)
			print('new delay-values: ', new_delay_values)

			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			if ( topology == 'square-periodic' or topology == 'square-open' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic'):
				Nx, Ny = get2DosciNumbers()										# calls function that asks user for input of number of oscis in each direction
				N = Nx*Ny
				kx, ky = get2DTwistNumbers(Nx, Ny, topology)					# choose 2d-twist under investigation
				k = kx															# set to dummy value
			else:
				N = chooseNumber()												# calls function that asks user for input of number of oscis
				k = chooseTwistNumber(N, topology)								# choose twist under investigation
				kx = k; ky = -999; Nx = N; Ny = 1;
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise (VCO)
			cLF		= chooseLF_DiffConst()										# calls function that asks user for input of diffusion constant GWN dynamic noise (LF)
			# pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			pert = []
			# Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations

			# if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
			# 	h = synctools.Triangle(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'cos':
			# 	h = synctools.Cos(1.0 / (2.0 * np.pi))
			# elif str(params['DEFAULT']['couplingfct']) == 'sin':
			# 	h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			isRadian=False
			sf = synctools.SweepFactory(N, Ny, Nx, F, K, new_delay_values, str(params['DEFAULT']['couplingfct']), Fc, k, kx, ky,topology, c, isRadians=isRadian)

			fsl = sf.sweep()
			para_mat_temp = fsl.get_parameter_matrix(isRadians=False)			# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {N, F, K, Fc, delay, m, F_Omeg, ReLambda, ImLambda, Tsim, Nx, Ny, mx, my}: \n', para_mat_temp)

			para_mat = chooseCsvSaveOption(param_cases_csv, para_mat_temp, topology, str(params['DEFAULT']['couplingfct']), c)

			if not para_mat == [] and not len(para_mat) == 0:
				if len(para_mat[:,0]) > 1:
					plot_out = False
				elif len(para_mat[:,0]) == 1:
					plot_out = True
				else:
					plot_out = False

				for i in range (len(para_mat[:,0])):
					print('Tsim: ', para_mat[i,9])
					print('\nSTART: python case_bruteforce.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '
													+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '
													+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+' '+str(Nx)+' '+str(Ny)+' '+str(kx)+' '+str(ky)+' '+str(cLF)+' '
													+' '.join(map(str, pert)))
					# os.system('python case_bruteforce.py '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(para_mat[i,9]))))+' '+str(c)+' '+'1'+str(Nx)+str(Ny)+str(kx)+str(ky)+' '+' '.join(map(str, pert)))
					# print('\ncall singleout from guided_execution with: ', str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]), int(para_mat[i,5]), int(round(float(para_mat[i,9]))), c, 1, pert, '\n')
					cbrut.bruteforceout(str(topology), int(para_mat[i,0]), float(para_mat[i,2]), float((para_mat[i,3])), float((para_mat[i,4])), float(para_mat[i,6]),
									int(para_mat[i,5]), int(round(float(para_mat[i,9]))), float(c), float(cLF), int(1), int(Nx), int(Ny), int(kx), int(ky),
									pert, plot_out)
			break

		else:
			print('Please provide input from the following options: K, Fc , delay in [s]')

	return None

''' MAIN '''
if __name__ == '__main__':
	''' MAIN: organizes the execution of the DPLL simulation modes '''

	# load parameter param_cases_csv from file, specify delimiter and which line contains the colum description
	param_cases_csv = pd.read_csv('GlobFreq_LinStab/DPLLParameters.csv', delimiter=",", header=2, dtype={'K': np.float, 'Fc': np.float, 'delay': np.float, 'F_Omeg': np.float,
	 								'm': np.int, 'Tsim': np.int, 'id': np.int, 'ReLambda': np.float, 'SimSeconds': np.float, 'topology': np.str, 'c': np.float, 'N': np.int, 'coupFct': np.str,
									'Nx': np.int, 'Ny': np.int, 'mx': np.int, 'my': np.int})#, skipinitialspace=True)
	# load the configuration parameters
	''' DATA CONTAINER NUMBER ONE '''
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
	# Tsim				= float(params['DEFAULT']['Tsim'])						# 9 the simulation time for each realization -- load above from csv

	a_true = True
	while a_true:
		# option to reset the configuration
		# inputstr = ('\nWould you like to reset one of the following system parameters: \n{\nenable multiprocessing [%s], \nnumber of available cores to use [%d], \nparameter discretization for brute force method [%d], \ntype of coupling fct [%0.2f], \nmean intrinsic frequencies in Hz [%0.2f], \nsample frequency time series in Hz  [%0.2f], \nstandard deviation intrinsic frequency [%0.2f], \nstandard deviation coupling strength [%0.5f]\n} \n\n[y]es/[n]o: '%(str(multiproc), int(numberCores), int(paramDiscretization), str(couplingfct), float(F), float(Fsim), float(domega), float(diffconstK))
		# print(inputstr)
		decision1 = raw_input('\nWould you like to reset one of the following system parameters: \n{\nenable multiprocessing, \nnumber of available cores to use, \nparameter discretization for brute force method, \ntype of coupling fct, \nintrinsic frequencies, \nsample frequency time series, \nstandard deviation intrinsic frequency, \nstandard deviation coupling strength\n} \n\n[y]es/[n]o: ')
		if decision1 == 'y':
			multiproc 			= raw_input('Multiprocessing choose [True] / [False] (string): ')
			if multiproc == 'True':
				numberCores = raw_input('How many cores are maximally available? [integer]: ')
			else:
				numberCores = 1
			paramDiscretization = raw_input('Number of samples points (for brute force basin stability) along each dimension in (0,2pi], provide [int]: ')
			couplingfct 		= raw_input('Choose coupling function from {sin, cos, triang} [string]: ')
			F 					= raw_input('Choose mean intrinsic frequency [float]: ')
			Fsim 				= raw_input('Choose sample frequency for the signal in multiples of the intrinsic frequency [float]: ')
			domega     			= raw_input('Choose diffusion-constant for (static) distribution of intrinsic frequencies, zero implies identical frequencies [float]: ')
			diffconstK 			= raw_input('Choose diffusion-constant for (static) distribution of coupling strengths, zero implies identical coupling strength [float]: ')
			Tsim				= raw_input('Choose simulation time Tsim [float]: ')
			# write the new values to the file 1params.txt
			config = configparser.ConfigParser(allow_no_value = True)
			# config.add_section('DEFAULT')
			config.set('DEFAULT', '#1: for multiprocessing set to True')
			config.set('DEFAULT', 'multiproc', multiproc)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#2: discretization for brute force scan in rotated phase space with phi_k_prime (for each dimension)')
			config.set('DEFAULT', 'paramDiscretization', paramDiscretization)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#3: number of processors that can be used --> with INTEL HT: 2*number of cores of proc.')
			config.set('DEFAULT', 'numberCores', numberCores)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#4: type of coupling function -- choose from {triang(ular), cos(ine), sin(e)}')
			config.set('DEFAULT', 'couplingfct', couplingfct)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#5: free-running frequency of the PLLs [Hz]')
			config.set('DEFAULT', 'F', F)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#6: sample phases with Fsim [Hz] multiples of the intrinsic frequency -> goal: about 100 samples per period')
			config.set('DEFAULT', 'Fsim', Fsim)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#7: diffusion constant [variance=2*diffconst] of the gaussian distribution for the intrinsic frequencies')
			config.set('DEFAULT', 'domega', domega)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#8: diffusion constant [variance=2*diffconst] of the gaussian distribution for the coupling strength')
			config.set('DEFAULT', 'diffconstK', diffconstK)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#9: multiples of the period of the uncoupled oscillators to set simulation time -- usually set with argument on startup')
			config.set('DEFAULT', 'Tsim', Tsim)
			config.set('DEFAULT', '#')

			with open('1params.txt', 'w') as configfile:						# rewrite the 1params.txt file with the newly set values
				config.write(configfile)

			params.read('1params.txt')											# reload the 1params.txt file with the newly set values to set params-container
			multiproc 			= str(params['DEFAULT']['multiproc'])			# 1 for multiprocessing set to True (brute-force mode 2)
			paramDiscretization = int(params['DEFAULT']['paramDiscretization'])	# 2 discretization for brute force scan in rotated phase space with phi'_k
			numberCores 		= int(params['DEFAULT']['numberCores'])			# 3 number of child processes in multiproc mode
			couplingfct 		= str(params['DEFAULT']['couplingfct'])			# 4 coupling function for FokkerPlanckEq mode 3: {sin, cos, triang}
			F 					= float(params['DEFAULT']['F'])					# 5 free-running frequency of the PLLs
			Fsim 				= float(params['DEFAULT']['Fsim'])				# 6 simulate phase model with given sample frequency -- goal: about 100 samples per period
			domega     			= float(params['DEFAULT']['domega'])			# 7 the diffusion constant [variance=2*diffconst] of the gaussian distribution for the intrinsic frequencies
			diffconstK 			= float(params['DEFAULT']['diffconstK'])		# 8 the diffusion constant [variance=2*diffconst] of the gaussian distribution for the coupling strength
			# Tsim				= float(params['DEFAULT']['Tsim'])				# 9 the simulation time for each realization -- load above from csv
			break
		elif decision1 == 'n':
			params.read('1params.txt')											# reload the 1params.txt file with the newly set values to set params-container
			multiproc 			= str(params['DEFAULT']['multiproc'])			# 1 for multiprocessing set to True (brute-force mode 2)
			paramDiscretization = int(params['DEFAULT']['paramDiscretization'])	# 2 discretization for brute force scan in rotated phase space with phi'_k
			numberCores 		= int(params['DEFAULT']['numberCores'])			# 3 number of child processes in multiproc mode
			couplingfct 		= str(params['DEFAULT']['couplingfct'])			# 4 coupling function for FokkerPlanckEq mode 3: {sin, cos, triang}
			F 					= float(params['DEFAULT']['F'])					# 5 free-running frequency of the PLLs
			Fsim 				= float(params['DEFAULT']['Fsim'])				# 6 simulate phase model with given sample frequency -- goal: about 100 samples per period
			domega     			= float(params['DEFAULT']['domega'])			# 7 the diffusion constant [variance=2*diffconst] of the gaussian distribution for the intrinsic frequencies
			diffconstK 			= float(params['DEFAULT']['diffconstK'])		# 8 the diffusion constant [variance=2*diffconst] of the gaussian distribution for the coupling strength
			# Tsim				= float(params['DEFAULT']['Tsim'])				# 9 the simulation time for each realization -- load above from csv
			break
		else:
			print('Please provide either [y]es or [N]o!')

	# first ask whether singleout, brute-force, or noisy simulations
	# sim_mode = raw_input...
	a_true = True
	while a_true:
		decision2 = raw_input('\nPlease specify simulation mode: \n\n[1] single realization, \n[2] statistics on noisy realizations, \n[3] brute-force basin of attraction scan, \n[4] single realization, adiabatic change of cLF. \n\nChoice: ')
		if decision2 == '1':
			singleRealization(params)
			break
		elif decision2 == '2':
			noisyStatistics(params)
			break
		elif decision2 == '3':
			bruteForce(params, param_cases_csv)
			break
		elif decision2 == '4':
			singleAdiabatChange(params)
			break
		else:
			print('Please provide numbers 1 to 3 as an input!')
