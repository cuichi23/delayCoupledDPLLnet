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

import case_noisy as cnois
import case_bruteforce as cbrut
import case_singleout as csing

sys.path.append(os.path.abspath("./GlobFreq_LinStab"))
import synctools

''' NOTE
    -> script is organized such that it is placed in a folder that is created in the directory that contains the results
    -> extracted plots will then be placed in a new folder with the name determined by the constant parameters and are named such that they have an order
'''

# def copyFiles(newfilename1, newfilename2, targetdir, resultsfolder, in_plot_name):
#     # loop over all files in the target folder and copy them to the resultsfolder, thereby renaming them such that they arrange in order
#     print('Copy to folder with name: ', resultsfolder, ' from folder with name: ', targetdir)
#     for file in os.listdir(targetdir):
#         if not file.endswith(".npz"):                                           # exclude data files
#             print('Copy file: ', targetdir+file)
#             if file.endswith(".png"):                                           # pick png plots
#                 if file.startswith("rot_red_PhaseSpace_lastR"):
#                     shutil.copy2(targetdir+file, resultsfolder+'/png/lastR/scatt_'+newfilename1+'.png')
#                 if file.startswith("rot_red_PhaseSpace_meanR"):
#                     shutil.copy2(targetdir+file, resultsfolder+'/png/meanR/scatt_'+newfilename2+'.png')
#                 if file.startswith("imshow_PhaseSpace_lastR"):
#                     shutil.copy2(targetdir+file, resultsfolder+'/png/lastR/imsh_'+newfilename1+'.png')
#                 if file.startswith("imshow_PhaseSpace_meanR"):
#                     shutil.copy2(targetdir+file, resultsfolder+'/png/meanR/imsh_'+newfilename2+'.png')
#                     # add parameter values to the png file
#                     font = ImageFont.truetype("Calibri.ttf", 256)
#                     img = Image.open(resultsfolder+'/png/meanR/imsh_'+newfilename2+'.png')
#                     draw = ImageDraw.Draw(img)
#                     draw.text((300, 100), in_plot_name, (0, 0, 0), font=font)
#                     img.save(resultsfolder+'/png/meanR/imsh_'+newfilename2+'.png')
#             else:
#                 if file.startswith("rot_red_PhaseSpace_lastR"):
#                     shutil.copy2(targetdir+file, resultsfolder+'/pdf_lastR/scatt_'+newfilename1+'.pdf')
#                 if file.startswith("rot_red_PhaseSpace_meanR"):
#                     shutil.copy2(targetdir+file, resultsfolder+'/scatt_'+newfilename2+'.pdf')
#                 if file.startswith("imshow_PhaseSpace_lastR"):
#                     shutil.copy2(targetdir+file, resultsfolder+'/pdf_lastR/imsh_'+newfilename1+'.pdf')
#                 if file.startswith("imshow_PhaseSpace_meanR"):
#                     shutil.copy2(targetdir+file, resultsfolder+'/imsh_'+newfilename2+'.pdf')
#     return 0

def chooseTopology(None):
	a_true = True
	while a_true:
		# get user input to know which topology should be analyzed
		topology = raw_input("\nPlease specify topology from 1-dim [chain, ring], 2-dim [square, hexgon, octagon] or mean-field [global] to be analyzed: ")
		if ( topology == 'square' or topology == 'hexagon' or topology == 'octagon' or topology == 'chain' or topology == 'ring' or topology == 'global' )
			break
		else:
			print('Please provide one of these inputs [chain, ring, square, hexgon, octagon, global]!')

	return topology

def singleRealization(params):
	a_true = True
	while a_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input("\nPlease specify parameter {K in [Hz], Fc in [Hz], delay in [s]} whose dependencies to be analyzed: ")
		if user_input == 'K':
			user_sweep_start = float(raw_input("Please specify the range in which K should be simulated, start K_s in [Hz] = "))
			user_sweep_end	 = float(raw_input("Please specify the range in which K should be simulated, end K_e in [Hz] = "))
			user_sweep_discr = float(raw_input("Please specify the discretization steps in [Hz] dK = "))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			Fc    = float(raw_input("for fixed/constant cut-off frequency Fc in (0, inf) [Hz], choose value: "))
			delay = float(raw_input("for fixed/constant delay-value in [0, 5 * T_omega] [s], choose value: "))
			new_K_set = []
			topology = chooseTopology()											# calls function that asks user for input of type of network

			TODO: call Daniel's code to get the Omegas and perturbation-decay values for the entire sweep region Fomeg.append... Tsim.append(int(round(-25 / reallambda)))

			for i in range (len(new_K_values)):
				# new_K_set.append([new_K_values[i], Fc, delay])
				# print('New K set:', new_K_set)

				TODO: find solution for N>3 to provide the initial delta-perturbation automatically, user should be asked after supplying N, which initial perturbation he wants!
						CHECK, wether in the case of singleout and noisy the phase space rotation is eliminated!!!!!!!!!!!!

				os.system('python case_singleout', str(topology), str(N), str(new_K_values[i]), str(Fc), str(delay), str(Fomeg), str(m), str(Tsim), str(c), '1', Lösung für Störungen... )

			TODO) change output class, such that all data has added the values of the parameters K, Fc, tau, ...?

		elif user_input == 'Fc':
		elif user_input == 'delay':
		else:
			print('Please provide input from the following options: K, Fc , delay in [s]')

	return None

def noisyStatistics(params):
	a_true = True
	while a_true:
		# run simulations on local machine or on pks-queue
		queue_or_local = raw_input("\nRun simulations locally or on the PKS queuing system? [l]ocal / [q]ueue-pks: ")
		if ( queue_or_local == 'l' or queue_or_local == 'q' ):
			pass
			break
		else:
			print('Please provide input [l] for local or [q] for queue!')

	a_true = True
	while a_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input("\nPlease specify parameter {K in [Hz], Fc in [Hz], delay in [s]} whose dependencies to be analyzed: ")
		if user_input == 'K':
			user_sweep_start = float(raw_input("Please specify the range in which K should be simulated, start K_s in [Hz] = "))
			user_sweep_end	 = float(raw_input("Please specify the range in which K should be simulated, end K_e in [Hz] = "))
			user_sweep_discr = float(raw_input("Please specify the discretization steps in [Hz] dK = "))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			Fc    = float(raw_input("for fixed/constant cut-off frequency Fc in (0, inf) [Hz], choose value: "))
			delay = float(raw_input("for fixed/constant delay-value in [0, 5 * T_omega] [s], choose value: "))
			new_K_set = []
			topology = chooseTopology()											# calls function that asks user for input of type of network

			TODO: call Daniel's code to get the Omegas and perturbation-decay values for the entire sweep region Fomeg.append... Tsim.append(int(round(-25 / reallambda)))

			for i in range (len(new_K_values)):
				# new_K_set.append([new_K_values[i], Fc, delay])
				# print('New K set:', new_K_set)
				if queue_or_local == 'l':
					os.system('python case_noisy', str(topology), str(N), str(new_K_values[i]), str(Fc), str(delay), str(Fomeg), str(m), str(Tsim), str(c), str(Nsim))
				elif queue_or_local == 'q':
					# check whether in PKS network, establish connection if away?, browse into right folder, write the configuration, start the program in the queuing-system with qsub,
					# however, there must be a list with the parameter combinations supplied or so
					print('Sorry, not yet implemented! Abort!')
					pass

				TODO) change output class, such that all data has added the values of the parameters K, Fc, tau, ...?

		elif user_input == 'Fc':
		elif user_input == 'delay':
		else:
			print('Please provide input from the following options: K, Fc , delay in [s]')

	return None

def bruteForce(params):
	a_true = True
	while a_true:
		# run simulations on local machine or on pks-queue
		queue_or_local = raw_input("\nRun simulations locally or on the PKS queuing system? [l]ocal / [q]ueue-pks: ")
		if queue_or_local == 'l':
			# start all new combinations locally, after checking which already exist in the csv file
			pass
			break
		elif queue_or_local == 'q':
			# start all new combinations of parameters on the pks-queue after checking whether logged on to pks network
			print('\n\nSorry, not yet implemented!')
			pass
			break
		else:
			print('Please provide input [l] for local or [q] for queue!')


	a_true = True
	while a_true:
		# get user input to know which parameter should be analyzed
		user_input = raw_input("\nPlease specify parameter {K, Fc, delay} whose dependencies to be analyzed: ")
		if user_input == 'K':
			user_sweep_start = float(raw_input("Please specify the range in which K should be simulated, start K_s = "))
			user_sweep_end	 = float(raw_input("Please specify the range in which K should be simulated, end K_e = "))
			user_sweep_discr = float(raw_input("Please specify the discretization steps dK = "))
			new_K_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)

			Fc    = float(raw_input("for fixed/constant cut-off frequency in (0, inf), choose value: "))
			delay = float(raw_input("for fixed/constant delay-value in [0, 5 * T_omega], choose value: "))
			# find all lines with above fixed Parameters and sort them by value
			K_set  = data.loc[(data['delay']==delay) & (data['Fc']==Fc)].sort('K')
			# print('existing parameter sets: ', K_set)
			new_K_set = []
			for i in range (len(new_K_values)):
				new_K_set.append([new_K_values[i], Fc, delay])

			print('New K set:', new_K_set)
			# determine the set of parameter combinations that has not yet been simulated and create a list of those, substract the lines that already exist
			clean_K_set = np.setdiff1d(K_set, new_K_set)

''' MAIN '''
if __name__ == '__main__':
	''' MAIN: organizes the execution of the DPLL simulation modes '''

	# load parameter data from file, specify delimiter and which line contains the colum description
	data = pd.read_csv('GlobFreq_LinStab/DPLLParameters.csv', delimiter=",", header=2, dtype={'K': np.float, 'Fc': np.float, 'delay': np.float, 'FOmeg': np.float, 'k': np.int, 'Tsim': np.int, 'sim-time-approx': np.float})
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
		decision1 = raw_input("\nWould you like to reset the following system parameters: {enable multiprocessing, number of availibe cores to use, parameter discretization for brute force method, type of coupling fct, intrinsic frequency, sample frequency, standard deviation intrinsic frequency, standard deviation coupling strength} [y/N]: ")
		if decision1 == 'y':
			multiproc 			= raw_input("Multiprocessing choose True/False [string]: ")
			if multiproc == 'True':
				numberCores = raw_input("How many cores are maximally availible? [int]: ")
			else:
				numberCores = 1
			paramDiscretization = raw_input("Number of samples points along each dimension in (0,2pi], provide [int]: ")
			couplingfct 		= raw_input("Choose coupling function from {sin, cos, triang} [string]: ")
			F 					= raw_input("Choose mean intrinsic frequency [float]: ")
			Fsim 				= raw_input("Choose sample frequency for the signal [float]: ")
			domega     			= raw_input("Choose diffusion-constant for distribution of intrinsic frequencies, zero implies identical frequencies [float]: ")
			diffconstK 			= raw_input("Choose diffusion-constant for distribution of coupling strengths, zero implies identical coupling strength [float]: ")
			Tsim				= raw_input("Choose simulation time Tsim [float]: ")
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
		decision2 = raw_input("\nPlease specify simulation mode: [1] single realization, [2] statistics on noisy realizations, [3] brute-force basin of attraction scan. Choice: ")
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

			# call Daniels program to obtain Fomeg and the lambdas, + stability, etc...
			# # Perform a delay sweep
			# sf = synctools.SweepFactory(n, w, k, tau, h, wc, m)
			# fsl = sf.sweep()
			#
			# # Extract variables from the sweep
			# para_mat = fsl.get_parameter_matrix()
			# tau2 = fsl.get_tau()
			# omega2 = fsl.get_omega()
			# l2 = fsl.get_l()


			# # create a folder with name given by the constant parameters
			# resultsfolder = 'Kvar-Fc%0.2f-delay%0.2f'%(Fc, delay)
			# try:
			# 	os.makedirs(resultsfolder)
			# 	os.makedirs(resultsfolder+'/png/meanR/')
			# 	os.makedirs(resultsfolder+'/png/lastR/')
			# 	os.makedirs(resultsfolder+'/pdf_lastR/')
		    # except OSError as exception:
			# 	if exception.errno != errno.EEXIST:
			# 		raise
			# # loop trough all lines, extract id and use that to collect the data, i.e., copy all picturefiles into the
			# # folder created above and name them by the folder name with 'var' being the variable value from the respective line
			# for index, row in K_set.iterrows():
			# 	targetdir = '../%d/results/'%(int(K_set.loc[index, 'id']))              # the directory from which the results are extracted
			# 	newfilename1 = 'K_%0.2f_lR_id%d'%(K_set.loc[index, 'K'], int(K_set.loc[index, 'id'])) # the new file name for the result with the last R
			# 	newfilename2 = 'K_%0.2f_mR_id%d'%(K_set.loc[index, 'K'], int(K_set.loc[index, 'id'])) # the new file name for the result with the mean over R within 2 eigenperiods
			# 	in_plot_name = 'K=%0.2f Hz, Fc=%0.2f Hz, delay=%0.2f s'%(K_set.loc[index, 'K'],K_set.loc[index, 'Fc'],K_set.loc[index, 'delay'])
			# 	#oldfilenames =  listdir('../%d/results/'%(int(K_set.loc[index, 'id'])))
			# 	copyFiles(newfilename1, newfilename2, targetdir, resultsfolder, in_plot_name)


			break
		elif user_input == 'Fc':
			pass
			break
		elif user_input == 'delay':
			pass
			break
		else:
			print('Please provide on of the following inputs: [K] for coupling strength, [Fc] for cut-off frequency, or [delay] for transmission delay!')








	if user_input == "Fc":
		user_sweep_start = raw_input("Please specify the range in which K should be simulated, start Fc_s = ")
		user_sweep_end	 = raw_input("Please specify the range in which K should be simulated, end Fc_e = ")
		user_sweep_discr = raw_input("Please specify the discretization steps dFc = ")
		new_Fc_values	 = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)
		# list all possible values of Fc {.,.,.,} and delay {.,.,.,}
		delay_values  = data.delay.unique()
		delay_values  = delay_values[~np.isnan(delay_values)]
		delay_values.sort()
		coupling_strengths = data.K.unique()
		coupling_strengths = coupling_strengths[~np.isnan(coupling_strengths)]
		coupling_strengths.sort()
		print('possible values of K = {', coupling_strengths, '}\n and delays = {', delay_values,'}')
		print('analyze Fc-dependency of basin-stability for fixed values of K and the delay:')
		K     = float(raw_input("for constant coupling strength: "))
		delay = float(raw_input("for constant delay-value: "))
		# find all lines which above fixed Parameters
		Fc_set  = data.loc[(data['delay']==delay) & (data['K']==K)].sort('Fc')
		# create a folder with name given by the constant parameters
		resultsfolder = 'Fcvar-K%0.2f-delay%0.2f'%(K, delay)
		try:
			os.makedirs(resultsfolder)
			os.makedirs(resultsfolder+'/png/meanR/')
			os.makedirs(resultsfolder+'/png/lastR/')
			os.makedirs(resultsfolder+'/pdf_lastR/')
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise
		# loop trough all lines, extract id and use that to collect the data, i.e., copy all picturefiles into the
		# folder created above and name them by the folder name with 'var' being the variable value from the respective line
		for index, row in Fc_set.iterrows():
			targetdir = '../%d/results/'%(int(Fc_set.loc[index, 'id']))
			newfilename1 = 'Fc_%0.2f_lR_id%d'%(Fc_set.loc[index, 'Fc'], int(Fc_set.loc[index, 'id']))
			newfilename2 = 'Fc_%0.2f_mR_id_%d'%(Fc_set.loc[index, 'Fc'], int(Fc_set.loc[index, 'id']))
			in_plot_name = 'Fc=%0.2f Hz, K=%0.2f Hz, delay=%0.2f s'%(Fc_set.loc[index, 'Fc'],Fc_set.loc[index, 'K'],Fc_set.loc[index, 'delay'])
			#oldfilenames =  listdir('../%d/results/'%(int(K_set.loc[index, 'id'])))
			copyFiles(newfilename1, newfilename2, targetdir, resultsfolder, in_plot_name)

	if user_input == "delay":
		user_sweep_start = raw_input("Please specify the range in which K should be simulated, start delay_s = ")
		user_sweep_end	 = raw_input("Please specify the range in which K should be simulated, end delay_e = ")
		user_sweep_discr = raw_input("Please specify the discretization steps ddelay = ")
		new_delay_values = np.arange(user_sweep_start, user_sweep_end + user_sweep_discr, user_sweep_discr)
		# list all possible values of Fc {.,.,.,} and delay {.,.,.,}
		cut_off_freqs = data.Fc.unique()
		cut_off_freqs = cut_off_freqs[~np.isnan(cut_off_freqs)]
		cut_off_freqs.sort()
		coupling_strengths = data.K.unique()
		coupling_strengths = coupling_strengths[~np.isnan(coupling_strengths)]
		coupling_strengths.sort()
		print('possible values of K = {', coupling_strengths, '}\n and Fc = {', cut_off_freqs,'}')
		print('analyze the delay-dependency of basin-stability for fixed values of K and Fc:')
		K  = float(raw_input("for constant coupling strength: "))
		Fc = float(raw_input("for constant cut-off frequency: "))
		# find all lines which above fixed Parameters
		delay_set  = data.loc[(data['K']==K) & (data['Fc']==Fc)].sort('delay')
		# create a folder with name given by the constant parameters
		resultsfolder = 'delayvar-Fc%0.2f-K%0.2f'%(Fc, K)
		try:
			os.makedirs(resultsfolder)
			os.makedirs(resultsfolder+'/png/meanR/')
			os.makedirs(resultsfolder+'/png/lastR/')
			os.makedirs(resultsfolder+'/pdf_lastR/')
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise
		# loop trough all lines, extract id and use that to collect the data, i.e., copy all picturefiles into the
		# folder created above and name them by the folder name with 'var' being the variable value from the respective line
		for index, row in delay_set.iterrows():
			targetdir = '../%d/results/'%(int(delay_set.loc[index, 'id']))
			newfilename1 = 'delay_%0.2f_lR_id%d'%(delay_set.loc[index, 'delay'], int(delay_set.loc[index, 'id']))
			newfilename2 = 'delay_%0.2f_mR_id%d'%(delay_set.loc[index, 'delay'], int(delay_set.loc[index, 'id']))
			in_plot_name = 'delay=%0.2f s, Fc=%0.2f Hz, K=%0.2f Hz'%(delay_set.loc[index, 'delay'],delay_set.loc[index, 'Fc'],delay_set.loc[index, 'K'])
			#oldfilenames =  listdir('../%d/results/'%(int(K_set.loc[index, 'id'])))
			copyFiles(newfilename1, newfilename2, targetdir, resultsfolder, in_plot_name)
