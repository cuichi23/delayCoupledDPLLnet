#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import evaluation as eva
import numpy as np
import numpy.ma as ma
import matplotlib
import pandas as pd
import csv
import sys
import os
if not os.environ.get('SGE_ROOT') == None:										# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.interpolate import spline
import math

import datetime
now = datetime.datetime.now()

''' All plots in latex mode '''
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rcParams['agg.path.chunksize'] = 10000

''' STYLEPACKS '''
titlefont = {
        'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 9,
        }

labelfont = {
        'family' : 'sans-serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 16,
        }

annotationfont = {
        'family' : 'monospace',
        'color'  : (0, 0.27, 0.08),
        'weight' : 'normal',
        'size'   : 14,
        }

''' EVALUATION OF DATA IN  1Anoisy_res.csv AND plotting'''
def loadData(head=1):
	''' loads the data from the results file for the statistics of noisy realizations, project with Shamik Gupta

		Author: Lucas Wetzel

	Parameters
	----------
	head		:	int
			 		number of header lines of csv file to load

	Variables
	---------
	meanRTsim	:	mean of last value of order parameter over all realizations
	stdRTsim	:	standard deviation of last value of order parameter over all realizations
	meanRtwoT	:	mean of last values (over 2 periods of the free-running frequency) of order parameter over all realizations
	stdRtwoT	:	standard deviation
	meanFTsim	:	mean of last instantaneous frequency of each realization and oscillator
	stdFTsim	:	standard deviation
	Nsim 		:	number of realizations

	Returns
	-------
	data		:	type:  <class 'pandas.core.frame.DataFrame'> --> np.array?
					results of simulations with PD noise (and potentially colored LF noise)'''


	print('Loading data-file results/1Anoisy_res.csv!')
	data = pd.read_csv('results/1Anoisy_res.csv', delimiter=",", header=head,
						dtype={'Nx': np.int, 'Ny': np.int, 'mx': np.int, 'my': np.int, 'K': np.float, 'Fc': np.float,
							   'delay': np.float, 'topology': np.str, 'F':np.float, 'F_Omeg': np.float, 'Tsim': np.int,
							   'Nsim': np.int, 'c': np.float, 'cPD': np.float, 'meanRTsim': np.float, '': np.float,
							   'stdRTsim': np.float, 'meanRtwoT': np.float, 'stdRtwoT': np.float, 'meanFTsim': np.float,
							   'stdFTsim': np.float})

	return data

def chooseK():																	# ask user-input for coupling strength
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		K = float(raw_input('\nPlease specify as [float] the coupling strength [Hz]: '))
		if K > 0:
			break
		else:
			print('Please provide input as an [float] in [0, oo]!')

	return float(K)

def chooseF():																	# ask user-input for intrinsic frequency
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		F = float(raw_input('\nPlease specify as [float] the intrinsic frequency F [Hz]: '))
		if F > 0:
			break
		else:
			print('Please provide input as an [float] in (0, oo] [Hz]!')

	return float(F)

def chooseFc():																	# ask user-input for cut-off frequency
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		Fc = float(raw_input('\nPlease specify as [float] the cut-off frequency Fc [Hz]: '))
		if Fc > 0:
			break
		else:
			print('Please provide input as an [float] in (0, oo] [Hz]!')

	return float(Fc)

def chooseMx():																	# ask user-input for twist number in x-direction
	a_true = True
	while a_true:
		# get user input on x-direction twist-number
		mx = int(raw_input('\nPlease specify as [integer] the x-direction twist number from [0, Nx-1]: '))
		if mx.is_integer:
			break
		else:
			print('Please provide input as an [integer] in (0, Nx-1], where Nx the number of oscillators in x-direction!')

	return int(mx)

def chooseDelay():																# ask user-input for delay
	a_true = True
	while a_true:
		# get user input on number of oscis in the network
		delay = float(raw_input('\nPlease specify as [float] the transmission delay [s]: '))
		if delay>=0:
			break
		else:
			print('Please provide input as an [float] in [0, 5 * T_intrinsic] [s]!')

	return float(delay)

def chooseTopology():															# ask user-input for topology
	a_true = True
	while a_true:
		# get user input to know which topology should be analyzed
		topology = str(raw_input('''\nPlease specify topology from
		1-dim [chain, ring],
		2-dim [square-open, square-periodic, hexagon, octagon, hexagon-periodic, octagon-periodic] or mean-field [global] to be analyzed:\n
		'''))
		if ( topology == 'square-open' or topology == 'square-periodic' or topology == 'hexagon' or topology == 'octagon' or topology == 'hexagon-periodic' or topology == 'octagon-periodic' or topology == 'chain' or topology == 'ring' or topology == 'global' ):
			break
		else:
			print('Please provide one of these input-strings: [chain, ring, square-open, square-periodic, hexagon, octagon, hexagon-periodic, octagon-periodic, global]!')

	return str(topology)

def extractData(odata, par, Nsim, Nx, F, Fc, K, delay, topology, mx):
	''' extracts from the data the results that are specified in the string 'par'

		Author: Lucas Wetzel

	Parameters
	----------
	Nsim		:	int
			 		number realizations computed
	Nx			:	int
					number of oscis in x-direction
	para		:	string {Fc, K, delay, topology}
					specifies which data needs to be extracted

	Returns
	-------
	sorted_data	:	type:  <class 'pandas.core.frame.DataFrame'>
					the data required to plot the results as specified by the user input '''

	err = float(0.01);																	# helper variable to do float extraction
	if par == 'Fc':
		temp_data = odata[(odata.Nx==Nx) & (odata.Nsim==Nsim) & (odata.topology==topology) & (odata.mx==mx)
						& (odata.K < (K + err)) & (odata.K > (K - err)) & (odata.delay < (delay + err)) & (odata.delay > (delay - err))
						& (odata.F < (F + err)) & (odata.F > (F - err))]
		# print('1: ', temp_data)

	elif par == 'delay':
		temp_data = odata[(odata.Nx==Nx) & (odata.Nsim==Nsim) & (odata.topology==topology) & (odata.mx==mx)
						& (odata.K < (K + err)) & (odata.K > (K - err)) & (odata.Fc< (Fc + err)) & (odata.Fc> (Fc - err))
						& (odata.F < (F + err)) & (odata.F > (F - err))]

	# print('\n\n type(temp_data): ', type(temp_data), 'length: ', len(temp_data))
	if len(temp_data) == 0:
		sys.exit("\n\nNO DATA AVAILABLE!\nTERMINATE\n")
	else:
		return temp_data

def plot_results(data, par, K, F, Fc, delay, topology, Nx, Nsim, mx):
	''' prints results for the statistics of noisy realizations, project with Shamik Gupta

		Author: Lucas Wetzel

	Parameters
	----------
	data		:	pd.DataFrame

	Variables
	---------

	Returns
	-------
	None		:	'''

	dpi_value = 300; err = 0.01; TOL = 10;
	''' Plot the mean over all realizations of the last order parameter value '''
	if par == 'Fc':
		plt.figure('R(tend) vs cPD')
		plt.clf()
		Fc_values = np.unique(data.Fc.round(decimals=TOL))						# extraxt the different values of the cut-off frequency, a plot will be made for each
		Fc_values.sort()														# sort values, smallest to largest
		for i in range(len(Fc_values)):

			td = data[(data.Fc < (Fc_values[i] + err)) & (data.Fc > (Fc_values[i] - err))]	# extract the rows of the data that all share the same cut-off frequency
			# print('td:\n',td)
			# plt.plot(td.cPD, td.mean_Rend, label=('Fc='+str(Fc_values[i])), marker='.')
			plt.errorbar(td.cPD, td.mean_Rend, td.std_Rend, label=('Fc='+str(Fc_values[i])), marker='.', capsize=3)

		plt.plot(td.cPD, np.zeros(len(td.cPD))+1/np.sqrt(Nx), label=(r'1/$\sqrt{N}$'))	# plot line at 1/sqrt(N) to show the expected order parameter in a desyncronized system of N oscis
		plt.xlabel(r'$c$ $[s]$', fontdict=labelfont)
		plt.ylabel(r'$R_{\rm Tsim}( t,m = %d )$' % mx, fontdict=labelfont)
		plt.legend(loc='best')
		# plt.savefig('results/orderP-vs-cPD_K%.2f_F%.2f_tau%.4f_%d_%d_%d.pdf' %(K, F, delay, now.year, now.month, now.day))
		plt.savefig('results/oPTsim-vs-cPD_K%.2f_F%.2f_tau%.4f_%d_%d_%d.png' %(K, F, delay, now.year, now.month, now.day), dpi=dpi_value)

		plt.figure('R(2T) vs cPD')
		plt.clf()
		for i in range(len(Fc_values)):

			td = data[(data.Fc < (Fc_values[i] + err)) & (data.Fc > (Fc_values[i] - err))]	# extract the rows of the data that all share the same cut-off frequency
			# print('td:\n',td)
			# plt.plot(td.cPD, td.mean_R_2Tomeg, label=('Fc='+str(Fc_values[i])), marker='.')
			plt.errorbar(td.cPD, td.mean_R_2Tomeg, td.std_R_2Tomeg, label=('Fc='+str(Fc_values[i])), marker='.', capsize=3)

		plt.plot(td.cPD, np.zeros(len(td.cPD))+1/np.sqrt(Nx), label=(r'1/$\sqrt{N}$'))	# plot line at 1/sqrt(N) to show the expected order parameter in a desyncronized system of N oscis
		plt.xlabel(r'$c$ $[s]$', fontdict=labelfont)
		plt.ylabel(r'$\bar{R}_{\rm 2T}( t,m = %d )$' % mx, fontdict=labelfont)
		plt.legend(loc='best')
		# plt.savefig('results/orderP-vs-cPD_K%.2f_F%.2f_tau%.4f_%d_%d_%d.pdf' %(K, F, delay, now.year, now.month, now.day))
		plt.savefig('results/oP2T-vs-cPD_K%.2f_F%.2f_tau%.4f_%d_%d_%d.png' %(K, F, delay, now.year, now.month, now.day), dpi=dpi_value)

	elif par == 'delay':
		plt.figure('R(t) vs cPD')
		plt.clf()
		delay_values = np.unique(data.delay.round(decimals=TOL))				# extraxt the different values of the cut-off frequency, a plot will be made for each
		delay_values.sort()														# sort values, smallest to largest
		for i in range(len(delay_values)):

			td = data[(data.delay < (delay_values[i] + err)) & (data.delay > (delay_values[i] - err))]	# extract the rows of the data that all share the same cut-off frequency
			plt.errorbar(td.cPD, td.mean_Rend, td.std_Rend, label=(r'$\tau$='+str(delay_values[i])), marker='.', capsize=3)

		plt.plot(td.cPD, np.zeros(len(td.cPD))+1/np.sqrt(Nx), label=(r'1/$\sqrt{N}$'))	# plot line at 1/sqrt(N) to show the expected order parameter in a desyncronized system of N oscis
		plt.xlabel(r'$c$ $[s]$', fontdict=labelfont)
		plt.ylabel(r'$R_{}( t,m = %d )$' % mx, fontdict=labelfont)
		plt.legend(loc='best')
		# plt.savefig('results/orderP-vs-cPD_K%.2f_F%.2f_Fc%.2f_%d_%d_%d.pdf' %(K, F, Fc, now.year, now.month, now.day))
		plt.savefig('results/orderP-vs-cPD_K%.2f_F%.2f_Fc%.2f_%d_%d_%d.png' %(K, F, Fc, now.year, now.month, now.day), dpi=dpi_value)

		plt.figure('R(2T) vs cPD')
		plt.clf()
		for i in range(len(delay_values)):

			td = data[(data.delay < (delay_values[i] + err)) & (data.delay > (delay_values[i] - err))]	# extract the rows of the data that all share the same cut-off frequency
			# print('td:\n',td)
			# plt.plot(td.cPD, td.mean_R_2Tomeg, label=('Fc='+str(Fc_values[i])), marker='.')
			plt.errorbar(td.cPD, td.mean_R_2Tomeg, td.std_R_2Tomeg, label=(r'$\tau$='+str(delay_values[i])), marker='.', capsize=3)

		plt.plot(td.cPD, np.zeros(len(td.cPD))+1/np.sqrt(Nx), label=(r'1/$\sqrt{N}$'))	# plot line at 1/sqrt(N) to show the expected order parameter in a desyncronized system of N oscis
		plt.xlabel(r'$c$ $[s]$', fontdict=labelfont)
		plt.ylabel(r'$\bar{R}_{\rm 2T}( t,m = %d )$' % mx, fontdict=labelfont)
		plt.legend(loc='best')
		# plt.savefig('results/orderP-vs-cPD_K%.2f_F%.2f_Fc%.2f_%d_%d_%d.pdf' %(K, F, Fc, now.year, now.month, now.day))
		plt.savefig('results/oP2T-vs-cPD_K%.2f_F%.2f_Fc%.2f_%d_%d_%d.png' %(K, F, Fc, now.year, now.month, now.day), dpi=dpi_value)

	plt.draw()
	plt.show()
	return None

''' MAIN '''
if __name__ == '__main__':
	''' MAIN: organizes the execution of the DPLL simulation modes

		Author: Lucas Wetzel

	Parameters
	----------
	Nsim		:	integer
					number of realizations, handed over on program call
	Nx			:	integer
					number of oscillators in x-direction
	TOL 		:   number of decimals taking into account on 'round' operation '''

	TOL     =   10;
	Nsim 	=	sys.argv[1];
	Nx		=	sys.argv[2];
	print('\nStarted plotting routine for Nx = ',Nx,' oscillators and Nsim = ',Nsim,' realizations.\n')
	pd.set_option('display.width', 220) 										# default is 80, sets how many columns are printed before a line-break
	data = loadData(1)															# one header line for the dimensions of the values
	# print('data container from csv-file has type: ', type(data))
	# print('\ntest: ', pd.DataFrame(data))
	# print('\ntest: ', pd.DataFrame(data, columns=['Nx', 'Ny']))
	# print('\ntest: ', data.Nx, data.Nx[3])

	a_true = True
	while a_true:
		# get user input on what shoud be plotted
		plotchoice = int(raw_input('''\nPlease specify by the indicated integers what to plot!\n
		case 0: plot order parameter vs cPD for different Fc (0)
		case 1: plot order parameter vs cPD for different delays (1)
		choice: '''))
		if ( plotchoice>=0 and plotchoice<=1 ):
			print('\n')
			break
		else:
			print('Please provide input as an [integer] in [0 1]')

	if plotchoice == 0:															# plot for different Fc

		print('List of Fc-values for which data is available. Fc ->',np.unique(data.Fc.round(decimals=TOL)))
		print('''For this choice, the parameters other than Fc, i.e., topology, K and delay, can have these values respectively:\n
		topology	-> ''',data.topology.unique(),'''
		K		-> ''',np.unique(data.K.round(decimals=TOL)),'''
		mx		-> ''',data.mx.unique(),'''
		F		-> ''',data.F.unique(),'''
		delay		-> ''',np.unique(data.delay.round(decimals=TOL)),'''\n''')
		if len(data.mx.unique()) > 1:
			mx = chooseMx()
		else:
			mx = int(data.mx.unique()[0])

		if len(np.unique(data.K.round(decimals=TOL))) > 1:
			K = chooseK()
		else:
			K = float(np.unique(data.K.round(decimals=TOL))[0])

		if len(np.unique(data.F.round(decimals=TOL))) > 1:
			F = chooseF()
		else:
			F = float(np.unique(data.F.round(decimals=TOL))[0])

		if len(np.unique(data.delay.round(decimals=TOL)))	> 1:
			delay = chooseDelay()
		else:
			delay = float(np.unique(data.delay.round(decimals=TOL))[0])

		if len(data.topology.unique()) > 1:
			topology = chooseTopology()
		else:
			topology = str(data.topology.unique()[0])

		extracted_data = extractData(data, 'Fc', int(Nsim), int(Nx), float(F), 0.0, float(K), float(delay), str(topology), int(mx))
		# print(extracted_data)
		plot_results(extracted_data, 'Fc', float(K), float(F), -23., float(delay), str(topology), int(Nx), int(Nsim), int(mx))

	elif plotchoice == 1:														# plot for different delay

		print('List of Fc-values for which data is available. Fc ->',data.delay.unique())
		print('''For this choice, the parameters other than delay, i.e., topology, K and Fc, can have these values respectively:\n
		topology	-> ''',data.topology.unique(),'''
		K		-> ''',data.K.unique(),'''
		mx		-> ''',data.mx.unique(),'''
		F		-> ''',data.F.unique(),'''
		Fc		-> ''',data.Fc.unique(),'''\n''')
		if len(data.mx.unique()) > 1:
			mx = chooseMx()
		else:
			mx = int(data.mx.unique()[0])

		if len(np.unique(data.K.round(decimals=TOL))) > 1:
			K = chooseK()
		else:
			K = float(np.unique(data.K.round(decimals=TOL))[0])

		if len(np.unique(data.F.round(decimals=TOL))) > 1:
			F = chooseF()
		else:
			F = float(np.unique(data.F.round(decimals=TOL))[0])

		if len(np.unique(data.Fc.round(decimals=TOL))) > 1:
			Fc = chooseFc()
		else:
			Fc = float(np.unique(data.Fc.round(decimals=TOL))[0])

		if len(data.topology.unique()) > 1:
			topology = chooseTopology()
		else:
			topology = str(data.topology.unique()[0])

		extracted_data = extractData(data, 'delay', int(Nsim), int(Nx), float(F), float(Fc), float(K), 0.0, str(topology), int(mx))
		plot_results(extracted_data, 'delay', float(K), float(F), float(Fc),-23., str(topology), int(Nx), int(Nsim), int(mx))

	else:
		print('Please make a choice from the options given.')
