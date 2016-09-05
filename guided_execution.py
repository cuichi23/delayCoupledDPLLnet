from __future__ import print_function
import os
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

import evaluation as eva

import case_noisy as cnois
import case_bruteforce as cbrut
import case_singleout as csing

sys.path.append(os.path.abspath('./GlobFreq_LinStab'))
import synctools

def chooseTopology():															# ask user-input for topology
	a_true = True
	while a_true:
		# get user input to know which topology should be analyzed
		topology = str(raw_input('\nPlease specify topology from 1-dim [chain, ring], 2-dim [square, hexgon, octagon] or mean-field [global] to be analyzed: '))
		if ( topology == 'square' or topology == 'hexagon' or topology == 'octagon' or topology == 'chain' or topology == 'ring' or topology == 'global' ):
			break
		else:
			print('Please provide one of these input-strings: [chain, ring, square, hexgon, octagon, global]!')

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

def chooseK(mean_int_freq):														# ask user-input for coupling strength
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		K = float(raw_input('\nPlease specify as [float] the coupling strength [Hz]: '))
		if np.abs( float(K) ) < 5.0 * mean_int_freq:
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

def chooseTransDelay():															# ask user-input for delay
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		delay = float(raw_input('\nPlease specify as [float] the transmission delay [s]: '))
		if float(delay)>0:
			break
		else:
			print('Please provide input as an [float] in [0, 5 * T_intrinsic] [s]!')

	return float(delay)

def chooseTwistNumber(N):														# ask user-input for delay
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		k = raw_input('\nPlease specify the m-twist number [integer] in [0, 1, ..., %d] [dimless]: ' %(N-1))
		if ( int(k)>=0 ):
			break
		else:
			print('Please provide input as an [integer] in [0, %d]!' %(N-1))

	return int(k)

def chooseDeltaPert(N):															# ask user-input for delta-perturbation
	b_true = True
	while b_true:
		# get user input on delta-perturbation type
		choice_history = raw_input('\nPlease specify type delta-perturbation from {[1] manual, [2] all zero, [3] 1 of N perturbed, or [4] all perturbed according to iid distribution around zero}: ')
		if ( type(choice_history) == str and int(choice_history) > 0):
			pert = setDeltaPertubation(N, choice_history)				# sets the delta perturbation, ATTENTION: should here be given in rotated phase space
			break
		else:
			print('Please provide integer input from [1, 2, 3,...]!')

	return pert

def chooseDiffConst():															# ask user-input for diffusion constant GWN dynamic noise process
	b_true = True
	while b_true:
		# get user input on dynamic frequency noise -- provide diffusion constant for GWN process -> results in Wiener process for the phases (integrated instantaneous freqs)
		c = raw_input('\nPlease specify the diffusion constant [float] for the GWN process on the frequencies of the DPLLs from the range [0, 10*K]: ')
		if float(c) >= 0.0:
			break
		else:
			print('Please provide [float] input from [0, 10*K]!')

	return c

def chooseNsim():																# ask user-input for number of realizations for the noisy statistics
	b_true = True
	while b_true:
		# get user input on dynamic frequency noise -- provide diffusion constant for GWN process -> results in Wiener process for the phases (integrated instantaneous freqs)
		Nsim = raw_input('\nPlease specify how many realizations [integer] of the noisy dynamics should be simulated and averaged from [1, 1E5]: ')
		if int(Nsim) >= 1:
			break
		else:
			print('Please provide [integer] input from [1, 1E5]!')

	return Nsim

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

def setDeltaPertubation(N, case):
	a_true = True
	while a_true:
		# get user input on whether to input perturbation in rotated or original phase space
		rot_vs_orig = raw_input('\nPlease specify whether perturbation is provided in rotated [r] or original [o] phase space: ')
		if rot_vs_orig == 'r':													# perturbation provided in rotated phase space

			if case == '1':														# case in which the user provides all pertubations manually
				perturbation_vec = loopUserInputPert(N)
				break
			elif case == '2':													# case for which no delta-perturbations are added
				print('No delta-perturbations, all set to zero!')
				phiS = np.zeros(N)
				perturbation_vec = phiS											# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
				break
			elif case == '3':													# case in which only one the the oscillators is perturbed
				phiS = np.zeros(N);
				b_true = True
				while b_true:
					# get user input on number of oscis in the network
					number 	   = int(raw_input('\nWhich oscillator id [integer] in [1, %d] should be perturbed: ' %(N-1) ))
					pert_value = float(raw_input('Please provide perturbation as float in (0, 2pi] to be added to oscillator id=%d' %number))
					if (type(number) == int and number > 0  and number < N and pert_value>0 and pert_value<2.0*np.pi ):
						phiS[number] = pert_value
						break
					else:
						print('Please provide id as an [integer] in [1, %d] and the perturbation as a [float] in (0,2pi]!' %(N-1) )
				perturbation_vec = phiS											# no transformation necessary, since the modules expect to get the values in rotated phae space
				break
			elif case == '4':													# case in which all the oscillators are perturbed randomly from iid dist. in [min_pert, max_pert]
				# perturbation_vec =
				break

		elif rot_vs_orig == 'o':												# perturbation provided in original phase space

			if case == '1':														# case in which the user provides all pertubations manually
				phiS = loopUserInputPert(N)
				perturbation_vec = eva.rotate_phases(phiS, isInverse=True)		# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
				break
			elif case == '2':													# case for which no delta-perturbations are added
				print('No delta-perturbations, all set to zero!')
				phiS = np.zeros(N)
				perturbation_vec = eva.rotate_phases(phiS, isInverse=True)		# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
				break
			elif case == '3':													# case in which only one the the oscillators is perturbed
				phiS = np.zeros(N);
				b_true = True
				while b_true:
					# get user input on number of oscis in the network
					number 	   = int(raw_input('\nWhich oscillator id [integer] in [1, %d] should be perturbed: ' %(N-1) ))
					pert_value = float(raw_input('Please provide perturbation as float in (0, 2pi] to be added to oscillator id=%d: ' %number))
					if (type(number) == int and number > 0  and number < N and pert_value>0 and pert_value<2.0*np.pi ):
						phiS[number] = pert_value
						break
					else:
						print('Please provide id as an [integer] in [1, %d] and the perturbation as a [float] in (0,2pi]!' %(N-1) )
				perturbation_vec = eva.rotate_phases(phiS, isInverse=True)		# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
				break
			elif case == '4':													# case in which all the oscillators are perturbed randomly from iid dist. in [min_pert, max_pert]

				perturbation_vec = eva.rotate_phases(phiS, isInverse=True)		# transform perturbations into rotated phase space of phases, as required by case_noisy, case_singleout
				break

		else:
			print('Please provide one of these input-strings: [r] or [o] (small O)!')

	return perturbation_vec

def singleRealization(params):
	x_true = True
	while x_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input('\nPlease specify which parameters dependencies to be analyzed {[K] in [Hz], [Fc] in [Hz], [delay] in [s]} : ')
		if user_input == 'K':
			user_sweep_start = float(raw_input('\nPlease specify the range in which K should be simulated, start K_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end K_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dK = '))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			# Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a K sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*new_K_values, delay, h, 2.0*np.pi*Fc, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)

			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+'1'+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		elif user_input == 'Fc':
			user_sweep_start = float(raw_input('\nPlease specify the range in which Fc should be simulated, start Fc_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end Fc_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dFc = '))
			new_Fc_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			# Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a Fc sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*K, delay, h, 2.0*np.pi*new_Fc_values, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)

			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+'1'+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		elif user_input == 'delay':
			user_sweep_start = float(raw_input('\nPlease specify the range in which delays should be simulated, start delay_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end delay_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] ddelay = '))
			new_delay_values = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			# Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*K, new_delay_values, h, 2.0*np.pi*Fc, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)

			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+'1'+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		else:
			print('Please provide input from the following options: K, Fc , delay in [s]')

	return None

def noisyStatistics(params):
	x_true = True
	while x_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input('\nPlease specify which parameters dependencies to be analyzed {[K] in [Hz], [Fc] in [Hz], [delay] in [s]} : ')
		if user_input == 'K':
			user_sweep_start = float(raw_input('\nPlease specify the range in which K should be simulated, start K_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end K_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dK = '))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a K sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*new_K_values, delay, h, 2.0*np.pi*Fc, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)

			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+str(Nsim)+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		elif user_input == 'Fc':
			user_sweep_start = float(raw_input('\nPlease specify the range in which Fc should be simulated, start Fc_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end Fc_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dFc = '))
			new_Fc_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a Fc sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*K, delay, h, 2.0*np.pi*new_Fc_values, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)

			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+str(Nsim)+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		elif user_input == 'delay':
			user_sweep_start = float(raw_input('\nPlease specify the range in which delays should be simulated, start delay_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end delay_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] ddelay = '))
			new_delay_values = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*K, new_delay_values, h, 2.0*np.pi*Fc, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)

			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+str(Nsim)+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		else:
			print('Please provide input from the following options: K, Fc , delay in [s]')

	return None

def bruteForce(params):
	x_true = True
	while x_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input('\nPlease specify which parameters dependencies to be analyzed {[K] in [Hz], [Fc] in [Hz], [delay] in [s]} : ')
		if user_input == 'K':
			user_sweep_start = float(raw_input('\nPlease specify the range in which K should be simulated, start K_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end K_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dK = '))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			# Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a K sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*new_K_values, delay, h, 2.0*np.pi*Fc, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)


			ADD HERE TO COMPARE WHICH CASES ARE ALREADY IN THE CSV FILE :)
			K_set  = data.loc[(data['delay']==delay) & (data['Fc']==Fc)].sort('K')


			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+'1'+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		elif user_input == 'Fc':
			user_sweep_start = float(raw_input('\nPlease specify the range in which Fc should be simulated, start Fc_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end Fc_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] dFc = '))
			new_Fc_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			delay 	= chooseTransDelay()										# calls function that asks user for input of mean transmission delay
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			# Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a Fc sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*K, delay, h, 2.0*np.pi*new_Fc_values, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)

ADD HERE TO COMPARE WHICH CASES ARE ALREADY IN THE CSV FILE :)
			Fc_set  = data.loc[(data['delay']==delay) & (data['K']==K)].sort('Fc')

			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+'1'+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		elif user_input == 'delay':
			user_sweep_start = float(raw_input('\nPlease specify the range in which delays should be simulated, start delay_s in [Hz] = '))
			user_sweep_end	 = float(raw_input('\nPlease specify the range in which K should be simulated, end delay_e in [Hz] = '))
			user_sweep_discr = float(raw_input('\nPlease specify the discretization steps in [Hz] ddelay = '))
			new_delay_values = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			N 		= chooseNumber()											# calls function that asks user for input of number of oscis
			k		= chooseTwistNumber(N)										# choose twist under investigation
			K    	= chooseK(float(params['DEFAULT']['F']))					# calls function that asks user for input of the coupling strength
			Fc    	= chooseFc()												# calls function that asks user for input of cut-off frequency
			topology= chooseTopology()											# calls function that asks user for input of type of network topology
			c 		= chooseDiffConst()											# calls function that asks user for input of diffusion constant GWN dynamic noise
			pert 	= chooseDeltaPert(N)										# calls function that asks user for input for delta-like perturbation
			# Nsim    = chooseNsim()												# calls function that asks user for input for number of realizations
			if str(params['DEFAULT']['couplingfct']) == 'triang':				# set the coupling function for evaluating the frequency and stability with Daniel's module
				h = synctools.Triangle(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'cos':
				h = synctools.Cos(1.0 / (2.0 * np.pi))
			elif str(params['DEFAULT']['couplingfct']) == 'sin':
				h = synctools.Sin(1.0 / (2.0 * np.pi))
			print('params', params)

			# perform a delay sweep
			sf = synctools.SweepFactory(N, 2.0*np.pi*F, 2.0*np.pi*K, new_delay_values, h, 2.0*np.pi*Fc, k)
			fsl = sf.sweep()
			para_mat = fsl.get_parameter_matrix(isRadians=False)				# extract variables from the sweep, this matrix contains all cases
			print('New parameter combinations with {delay, K, Fc, F_Omeg, and Tsim} approximation:', para_mat)

ADD HERE TO COMPARE WHICH CASES ARE ALREADY IN THE CSV FILE :)
			delay_set  = data.loc[(data['K']==K) & (data['Fc']==Fc)].sort('delay')

			for i in range (len(para_mat)):
					print('python case_singleout '+str(topology)+' '+str(int(para_mat[i,0]))+' '+str(float(para_mat[i,2]))+' '+str(float((para_mat[i,3])))+' '+str(float(para_mat[i,4]))+' '+str(float(para_mat[i,6]))+' '+str(int(para_mat[i,5]))+' '+str(int(round(float(-25.0/para_mat[i,7]))))+' '+str(c)+' '+'1'+' '+' '.join(map(str, pert)))
					# os.system('python case_singleout '+str(topology)+''+str(N)+''+str(new_K_values[i])+''+str(Fc)+''+str(delay)+''+str(F_Omeg)+''+str(m)+''+str(Tsim)+''+str(c)+''+'1'+''+' '.join(map(str, pert)))
			break

		else:
			print('Please provide input from the following options: K, Fc , delay in [s]')

	return None

''' MAIN '''
if __name__ == '__main__':
	''' MAIN: organizes the execution of the DPLL simulation modes '''

	# load parameter data from file, specify delimiter and which line contains the colum description
	data = pd.read_csv('GlobFreq_LinStab/DPLLParameters.csv', delimiter=",", header=2, dtype={'K': np.float, 'Fc': np.float, 'delay': np.float, 'F_Omeg': np.float, 'k': np.int, 'Tsim': np.int, 'sim-time-approx': np.float})
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
		decision1 = raw_input('\nWould you like to reset the following system parameters: {enable multiprocessing, number of availibe cores to use, parameter discretization for brute force method, type of coupling fct, intrinsic frequency, sample frequency, standard deviation intrinsic frequency, standard deviation coupling strength} [y/N]: ')
		if decision1 == 'y':
			multiproc 			= raw_input('Multiprocessing choose True/False [string]: ')
			if multiproc == 'True':
				numberCores = raw_input('How many cores are maximally availible? [int]: ')
			else:
				numberCores = 1
			paramDiscretization = raw_input('Number of samples points along each dimension in (0,2pi], provide [int]: ')
			couplingfct 		= raw_input('Choose coupling function from {sin, cos, triang} [string]: ')
			F 					= raw_input('Choose mean intrinsic frequency [float]: ')
			Fsim 				= raw_input('Choose sample frequency for the signal [float]: ')
			domega     			= raw_input('Choose diffusion-constant for distribution of intrinsic frequencies, zero implies identical frequencies [float]: ')
			diffconstK 			= raw_input('Choose diffusion-constant for distribution of coupling strengths, zero implies identical coupling strength [float]: ')
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
			config.set('DEFAULT', '#5: free-running frequency of the PLLs')
			config.set('DEFAULT', 'F', F)
			config.set('DEFAULT', '#')
			config.set('DEFAULT', '#6: sample phases with Fsim multiples of the intrinsic frequency -> goal: about 100 samples per period')
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

			with open('1paramsTEST.txt', 'w') as configfile:					# rewrite the 1params.txt file with the newly set values
				config.write(configfile)

			params.read('1params.txt')											# reload the 1params.txt file with the newly set values to set params-container
			break
		elif decision1 == 'N':
			params.read('1params.txt')											# reload the 1params.txt file with the newly set values to set params-container
			break
		else:
			print('Please provide either [y]es or [N]o!')

	# first ask whether singleout, brute-force, or noisy simulations
	# sim_mode = raw_input...
	a_true = True
	while a_true:
		decision2 = raw_input('\nPlease specify simulation mode: [1] single realization, [2] statistics on noisy realizations, [3] brute-force basin of attraction scan. Choice: ')
		if decision2 == '1':
			singleRealization(params)
			break
		elif decision2 == '2':
			noisyStatistics(params)
			break
		elif decision2 == '3':
			bruteForce(params)
			break
		else:
			print('Please provide numbers 1 to 3 as an input!')