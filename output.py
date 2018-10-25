#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import evaluation as eva
import numpy as np
import numpy.ma as ma
import matplotlib
import csv
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

''' EVALUATION SINGLE REALIZATION '''
def plotTimeSeries(phi, F, Fc, dt, orderparam, k, delay, F_Omeg, K, c, cPD, cPD_t=[], Kadiab_t=[], K_adiab_r=(-1), coupFct='triang', Tsim=53, Fsim=None, show_plot=True):

	if np.size(Kadiab_t) == 0 and np.size(cPD_t) != 0:
		Trelax 			= Tsim
		Nrelax 			= int(Trelax/dt)
		rate   			= 0.1 / Trelax
		rate_per_step 	= 0.1 / Nrelax
		''' use Trelax to calculate the rate and Nsteps until cPD is close to zero '''
		Nsteps  	  	= int(0.95 * c / rate_per_step)
		delaysteps    	= int(delay / dt)
		Nsim   			= Nrelax+Nsteps+delaysteps
	elif np.size(Kadiab_t) != 0 and np.size(cPD_t) == 0:
		Trelax 			= Tsim
		Nrelax 			= int(Trelax/dt)
		rate   			= 0.1 / Trelax											# self.K_rate is set in simulation.py
		rate_per_step 	= 0.1 / Nrelax
		''' calculate the rate and Nsteps until K has grown and decayed back to its initial value '''
		Nsteps  	  	= 2 * int( (2.0*np.pi*(K_adiab_r - K)) / rate_per_step )
		delaysteps    	= int(delay / dt)
		Nsim   			= Nrelax+Nsteps+delaysteps

	# print('cPD_t:', cPD_t)
	if F > 0:																	# for f=0, there would otherwies be a float division by zero
		F1=F
	else:
		F1=.1
	# print('\n\nUncomment \"matplotlib.use(\'Agg\')\" in ouput.py to enable plotting figures to the desktop, DISABLE for queue-jobs!')
	phi = phi[:,:,:]; orderparam = orderparam[0,:]								# there is only one realization of interest -reduces dimension of phi array
	# afterTransients = int( round( 0.5*Tsim / dt ) )
	# phiSpect = phi[:,-afterTransients:,:]
	now = datetime.datetime.now()
	''' only plot power spetral density if there is no adiabatic change '''
	if np.size(cPD_t) == 0 and np.size(Kadiab_t) == 0:
		if coupFct == 'triang':
			print('Calculate spectrum for square wave signals. Fsim=%d' %Fsim)
			# f, Pxx_db = eva.calcSpectrum( (phiSpect), Fsim, 'square')				# calculate spectrum of signals, i.e., of this state
			f, Pxx_db = eva.calcSpectrum( (phi), Fsim, 'square')					# calculate spectrum of signals, i.e., of this state
		elif coupFct == 'triangshift':
			print('Calculate spectrum for ??? wave signals, here triang[x]+a*triang[b*x]. Fsim=%d' %Fsim)
			# f, Pxx_db = eva.calcSpectrum( (phiSpect), Fsim, 'square')				# calculate spectrum of signals, i.e., of this state
			f, Pxx_db = eva.calcSpectrum( (phi), Fsim, 'square')					# calculate spectrum of signals, i.e., of this state
		elif coupFct == 'sin':
			print('check that... sine coupFct only if cos and sin signal input')
			# f, Pxx_db = eva.calcSpectrum( (phiSpect), Fsim, 'sin')				# calculate spectrum of signals, i.e., of this state
			f, Pxx_db = eva.calcSpectrum( (phi), Fsim, 'sin')						# calculate spectrum of signals, i.e., of this state
		elif coupFct == 'cos':
			print('Calculate spectrum for cosinusoidal signals. Fsim=%d' %Fsim)
			# f, Pxx_db = eva.calcSpectrum( (phiSpect), Fsim, 'cos')				# calculate spectrum of signals, i.e., of this state
			f, Pxx_db = eva.calcSpectrum( (phi), Fsim, 'cos')						# calculate spectrum of signals, i.e., of this state
		elif coupFct == 'sincos':
			print('Calculate spectrum for mix sine and cosine signals. Fsim=%d' %Fsim)
			# f, Pxx_db = eva.calcSpectrum( (phiSpect), Fsim, 'cos')				# calculate spectrum of signals, i.e., of this state
			f, Pxx_db = eva.calcSpectrum( (phi), Fsim, 'sin')						# calculate spectrum of signals, i.e., of this state

		plt.figure('spectrum of synchronized state')							# plot spectrum
		plt.clf()
		for i in range (len(f)):
			plt.plot(f[i], Pxx_db[i], '-')
		plt.title('power spectrum', fontdict = titlefont)
		plt.xlim(0,F1+20*K);	#plt.ylim(-100,0);
		plt.xlabel('frequencies [Hz]', fontdict = labelfont); plt.ylabel('P [dB]', fontdict = labelfont)
		plt.grid()
		plt.savefig('results/powerdensity_dB_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/powerdensity_dB_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)

	phi = phi[0,:,:];															# from here on the phi array is reduced in dimension - realization 0 picked
	t = np.arange(phi.shape[0])													# plot the phases of the oscillators over time

	plt.figure('histogram of frequencies')										# plot a histogram of the frequencies of the oscillators over time
	plt.clf()
	lastfreqs = (np.diff(phi[-int(2*1.0/(F1*dt)):, :], axis=0).flatten()/(dt))
	plt.hist(lastfreqs, bins=np.linspace(2*np.pi*(F1-2.*abs(K)), 2*np.pi*(F1+2.*abs(K)), num=21), rwidth=0.75 )
	plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.xlim((2*np.pi*(F1-abs(K)), 2*np.pi*(F1+abs(K))))
	plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/(2.0*np.pi) ), fontdict = titlefont)
	plt.xlabel(r'$\dot{\phi}(-2T -> T_{end})$ $[rad/s]$', fontdict = labelfont)
	plt.ylabel(r'histogram', fontdict = labelfont)
	plt.savefig('results/freqhistK%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/freqhistK%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)

	if ( len(phi[0, :])>100 ):
		plt.figure('histogram of phases at TSim')								# plot a histogram of the frequencies of the oscillators over time
		plt.clf()
		phasesTSim = phi[-2, :].flatten(); maxPhaseTsim=phasesTSim.max(); minPhaseTsim=phasesTSim.min();
		plt.hist(lastfreqs, bins=np.linspace(2*np.pi*(minPhaseTsim), 2*np.pi*(maxPhaseTsim), num=21), rwidth=0.75 )
		# plt.xlim((2*np.pi*(), 2*np.pi*()))
		plt.title(r'mean phase [rad] $\bar{\phi}=$%.3f and std $\bar{\sigma}_{\phi}=$%.4f' %( np.mean(phasesTSim), np.std(phasesTSim) ), fontdict = titlefont)
		plt.xlabel(r'$\phi(t=TSim)$ $[rad]$', fontdict = labelfont)
		plt.ylabel(r'histogram', fontdict = labelfont)
		plt.savefig('results/phaseHistoTSim_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/phaseHistoTSim_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)


	plt.figure('phases over time')
	plt.clf()
	if len((t*dt))!=len(phi[:,0]):												# in case we cut out a piece of the phase time-series (see case_singleadiabatic.py etc)
		tcut = np.arange(0,length(phi[:,0])-1) 									# then t has to be recalculated
		tcut = tcut*dt;										 					# make new tcut vector and correct time for entries after cut
		time_after_cut = t[-int(20*1.0/(F1*dt))];
		tcut[-int(20*1.0/(F1*dt)):] = tcut[-int(20*1.0/(F1*dt)):] + time_after_cut
		plt.plot(tcut,phi)
	else:
		plt.plot((t*dt),phi)
	plt.plot(delay, phi[int(round(delay/dt)),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'time series phases, inst. freq: $\dot{\phi}_0(t_{start})=%.4f$, $\dot{\phi}_0(t_{end})=%.4f$  [rad Hz]' %( (phi[int(2*1.0/(F1*dt))][0]-phi[1][0])/(2*1.0/F1-dt), (phi[-4][0]-phi[-3-int(2*1.0/(F1*dt))][0])/(2*1.0/F1-dt) ), fontdict = titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
	plt.ylabel(r'$\phi(t)$', fontdict = labelfont)
	plt.savefig('results/phases-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/phases-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)
	print(r'frequency of zeroth osci at the beginning and end of the simulation:, freqStart=%.4f, freqEnd=%.4f  [rad Hz]' %( (phi[int(2*1.0/(F1*dt))][0]-phi[1][0])/(2*1.0/F1-dt), (phi[-4][0]-phi[-4-int(2*1.0/(F1*dt))][0])/(2*1.0/F1-dt) ) )
	print('last values of the phases:\n', phi[-3:,:])

	plt.figure('phases over time wrapped 2pi')
	plt.clf()
	plt.plot((t*dt),phi%(2.*np.pi))												#math.fmod(phi[:,:], 2.*np.pi))
	plt.plot(delay, phi[int(round(delay/dt)),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'time series phases, inst. freq: $\dot{\phi}_0(t_{start})=%.4f$, $\dot{\phi}_0(t_{end})=%.4f$  [rad Hz]' %( (phi[int(2*1.0/(F1*dt))][0]-phi[1][0])/(2*1.0/F1-dt), (phi[-4][0]-phi[-3-int(2*1.0/(F1*dt))][0])/(2*1.0/F1-dt) ), fontdict = titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
	plt.ylabel(r'$\phi(t)$', fontdict = labelfont)
	plt.savefig('results/phases2pi-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/phases2pi-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)

	plt.figure('REWORK --> check poincare sections!!!!!        phase configuration between oscis, phase plot, poincare sections')
	plt.clf()
	for i in range(len(phi[0,:])):
		plt.plot((t*dt),((phi[:,i]-phi[:,0]+np.pi)%(2.*np.pi))-np.pi)			#math.fmod(phi[:,:], 2.*np.pi))
	plt.plot(delay, ((phi[int(round(delay/dt)),0]-phi[int(round(delay/dt)),1]+np.pi)%(2.*np.pi))-np.pi, 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'time series phase differences, inst. freq: $\dot{\phi}_0(t_{start})=%.4f$, $\dot{\phi}_0(t_{end})=%.4f$  [rad Hz]' %( (phi[int(2*1.0/(F1*dt))][0]-phi[1][0])/(2*1.0/F1-dt), (phi[-4][0]-phi[-3-int(2*1.0/(F1*dt))][0])/(2*1.0/F1-dt) ), fontdict = titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
	plt.ylabel(r'$\phi_k(t)-\phi_0(t)$', fontdict = labelfont)
	plt.savefig('results/phaseConf-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/phaseConf-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)

	plt.figure('frequencies over time')											# plot the frequencies of the oscillators over time
	plt.clf()
	phidot = np.diff(phi, axis=0)/dt
	# tnew = np.linspace( t[0], t[-1] , int(len(t[0:-1])/100) )
	# power_smooth = spline(t[0:-1], phidot, tnew)
	# plt.plot((t[0:-1]*dt), phidot, tnew, power_smooth)
	plt.plot((t[0:-1]*dt), phidot)
	plt.plot(delay-dt, phidot[int(round(delay/dt)-1),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'mean frequency over last $2T$-eigenperiods $\dot{\bar{\phi}}=%.4f$ [rad Hz]' % np.mean(phidot[-int(round(2*1.0/(F1*dt))):, 0] ), fontdict = titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
	plt.ylabel(r'$\dot{\phi}(t)$ [rad Hz]', fontdict = labelfont)
	plt.savefig('results/freq-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/freq-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)

	# print('\n\ncPD_t:', cPD_t, '\n\n')
	if not np.size(cPD_t) == 0:
		plt.figure('order parameter over time, adiabatic change cPD or c')		# plot the order parameter in dependence of time
		cPD_t=np.array(cPD_t)
		cPD_t=cPD_t.flatten()
		plt.clf()
		plt.plot((t[0:(len(t)-1):10*int(1/dt)]*dt), orderparam[0:(len(t)-1):10*int(1/dt)])
		plt.plot(t*dt, cPD_t)
		plt.plot(delay, orderparam[int(round(delay/dt))], 'yo', ms=5)			# mark where the simulation starts
		plt.grid()
		plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
		plt.title(r'$\bar{R}(t_{end}-2T_{\omega}:t_{end})=$%.2f, and $\bar{\sigma}=$%.4f' %(np.mean(orderparam[-int(round(2*1.0/(F1*dt))):]), np.std(orderparam[-int(round(2*1.0/(F1*dt))):])), fontdict = titlefont)
		plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
		plt.ylabel(r'$R( t,m = %d )$' % k, fontdict = labelfont)
		plt.savefig('results/orderP-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/orderP-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)
		#print('\nlast entry order parameter: R-1 = %.3e' % (orderparam[-1]-1) )
		#print('\nlast entries order parameter: R = ', orderparam[-25:])

		if c==0 and cPD>0:
			plt.figure('order parameter vs adiabatic change cPD')				# plot the order parameter in dependence of cPD
			plt.clf()
			plt.plot(cPD_t[0:(len(t)-1):10*int(1/dt)], orderparam[0:(len(t)-1):10*int(1/dt)])
			plt.title(r'R(t) vs cPD(t) after Trelax, adiabatic-rate=%.4f' %(rate), fontdict = titlefont)
			plt.xlabel(r'$cPD(t)$', fontdict = labelfont)
			plt.ylabel(r'$R( t,m = %d )$' % k, fontdict = labelfont)
			plt.savefig('results/orderP-cPD_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
			plt.savefig('results/orderP-cPD_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)
		elif c>0 and cPD==0:
			plt.figure('order parameter vs adiabatic change c')					# plot the order parameter in dependence of c
			plt.clf()
			plt.plot(cPD_t[0:(len(t)-1):10*int(1/dt)], orderparam[0:(len(t)-1):10*int(1/dt)])
			plt.title(r'R(t) vs c(t) after Trelax, adiabatic-rate=%.4f' %(rate), fontdict = titlefont)
			plt.xlabel(r'$c(t)$', fontdict = labelfont)
			plt.ylabel(r'$R( t,m = %d )$' % k, fontdict = labelfont)
			plt.savefig('results/orderP-c_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
			plt.savefig('results/orderP-c_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)
	elif not np.size(Kadiab_t) == 0:
			plt.figure('order parameter over time, adiabatic change K')			# plot the order parameter in dependence of time
			Kadiab_t=np.array(Kadiab_t)
			Kadiab_t=Kadiab_t.flatten()
			plt.clf()
			plt.plot((t[0:(len(t)-1):10*int(1/dt)]*dt), orderparam[0:(len(t)-1):10*int(1/dt)])
			plt.plot(t*dt, Kadiab_t)
			plt.plot(delay, orderparam[int(round(delay/dt))], 'yo', ms=5)			# mark where the simulation starts
			plt.grid()
			plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
			plt.title(r'$\bar{R}(t_{end}-2T_{\omega}:t_{end})=$%.2f, and $\bar{\sigma}=$%.4f' %(np.mean(orderparam[-int(round(2*1.0/(F1*dt))):]), np.std(orderparam[-int(round(2*1.0/(F1*dt))):])), fontdict = titlefont)
			plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
			plt.ylabel(r'$R( t,m = %d )$' % k, fontdict = labelfont)
			plt.savefig('results/orderP-t_K0%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
			plt.savefig('results/orderP-t_K0%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)
			#print('\nlast entry order parameter: R-1 = %.3e' % (orderparam[-1]-1) )
			#print('\nlast entries order parameter: R = ', orderparam[-25:])

			plt.figure('order parameter vs adiabatic change in K')				# plot the order parameter in dependence of cPD
			plt.clf()
			plt.plot(Kadiab_t[0:(len(t)-1):10*int(1/dt)], orderparam[0:(len(t)-1):10*int(1/dt)])
			plt.title(r'R(t) vs K(t) after Trelax, adiabatic-rate=%.4f' %(rate), fontdict = titlefont)
			plt.xlabel(r'$K(t)$', fontdict = labelfont)
			plt.ylabel(r'$R( t,m = %d )$' % k, fontdict = labelfont)
			plt.savefig('results/orderP-Kt_K0%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
			plt.savefig('results/orderP-Kt_K0%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)
	else:
		plt.figure('order parameter over time')									# plot the order parameter in dependence of time
		plt.clf()
		plt.plot((t*dt), orderparam)
		plt.plot(delay, orderparam[int(round(delay/dt))], 'yo', ms=5)			# mark where the simulation starts
		plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
		plt.title(r'mean order parameter $\bar{R}=$%.2f, and $\bar{\sigma}=$%.4f' %(np.mean(orderparam[-int(round(2*1.0/(F1*dt))):]), np.std(orderparam[-int(round(2*1.0/(F1*dt))):])), fontdict = titlefont)
		plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
		plt.ylabel(r'$R( t,m = %d )$' % k, fontdict = labelfont)
		plt.savefig('results/orderP-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/orderP-t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=300)
		#print('\nlast entry order parameter: R-1 = %.3e' % (orderparam[-1]-1) )
		#print('\nlast entries order parameter: R = ', orderparam[-25:])

	plt.draw()
	if show_plot:
		plt.show()

''' EVALUATION BRUTE-FORCE BASIN OF ATTRACTION '''
def doEvalBruteForce(Fc, F_Omeg, K, N, k, delay, twistdelta, results, allPoints, initPhiPrime0, phiMr, paramDiscretization, delays_0, twistdelta_x, twistdelta_y, topology, show_plot=True):
	''' Here addtional output, e.g., graphs or matrices can be implemented for testing '''
	# we want to plot all the m-twist locations in rotated phase space: calculate phases, rotate and then plot into the results
	twist_points  = np.zeros((N, N), dtype=np.float)							# twist points in physical phase space
	twist_pointsR = np.zeros((N, N), dtype=np.float)							# twist points in rotated phase space
	alltwistP = []
	now = datetime.datetime.now()

	if F_Omeg > 0:																# for f=0, there would otherwies be a float division by zero
		F1=F_Omeg
	else:
		F1=1.1

	if N == 2:																	# this part is for calculating the points of m-twist solutions in the rotated space, they are plotted later
		d1=0; d2=1;
		pass
	if N == 3:
		d1=1; d2=2;
		for i in range (N):
			twistdelta = ( 2.0*np.pi*i/(1.0*N) )
			twist_points[i,:] = np.array([0.0, twistdelta, 2.0*twistdelta]) 	# write m-twist phase configuation in phase space of phases
			# print(i,'-twist points:\n', twist_points[i,:], '\n')
			for m in range (-2,3):
				for n in range (-2,3):
					vtemp = twist_points[i,:] + np.array([0.0, 2.0*np.pi*m, 2.0*np.pi*n])
					alltwistP.append(vtemp)
					# print('vtemp:', vtemp, '\n')

		alltwistP = np.array(alltwistP)
		# print('alltwistP:\n', alltwistP, '\n')
		alltwistPR= np.transpose(eva.rotate_phases(np.transpose(alltwistP), isInverse=True))	# express the points in rotated phase space
		# print('alltwistP rotated (alltwistPR):\n', alltwistPR, '\n')

	''' PLOTS '''
	dpi_value = 300
	# print('\n\nUncomment \"matplotlib.use(\'Agg\')\" in ouput.py to enable plotting figures to the desktop, DISABLE for queue-jobs!')
	# "export DISPLAY=:99.0"
	# "sh -e /etc/init.d/xvfb start"
	# sleep 3 # give xvfb some time to start

	# scipy.interpolate.interp2d NOTE

	# print('results:', results.shape, '\n', results)
	# print('\nallPoints:', allPoints.shape, '\n', allPoints)
	# print('\nresults[:,0]:', results[:,0])
	# print('\nallPoints[:,0]+phiMr[d1]:', allPoints[:,0]+phiMr[d1])
	# print('\nallPoints[:,1]+phiMr[d2]:', allPoints[:,0]+phiMr[d1])
	''' prepare colormap for scatter plot that is always in [0, 1] or [min(results), max(results)] '''

	cdict = {
	  'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
	  'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
	  'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
	}

	colormap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

	''' IMPORTANT: since we add the perturbations additively, here we need to shift allPoints around the initial phases of the respective m-twist state, using phiMr '''
	plt.figure(1)																# plot the mean of the order parameter over a period 2T
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	plt.scatter(allPoints[:,0]+phiMr[d1], allPoints[:,1]+phiMr[d2], c=results[:,0], alpha=0.5, edgecolor='', cmap=colormap, vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
	if N==3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N==2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if N==3 and topology != "square-open" and topology != "chain":
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
	plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
	plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/rot_red_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/rot_red_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(2)
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	plt.scatter(allPoints[:,0]+phiMr[d1], allPoints[:,1]+phiMr[d2], c=results[:,1], alpha=0.5, edgecolor='', cmap=colormap, vmin=0, vmax=max(results[:,1]))
	plt.title(r'last $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
	if N==3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N==2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if N==3 and topology != "square-open" and topology != "chain":
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
	plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
	plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/rot_red_PhSpac_lastR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/rot_red_PhSpac_lastR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(3)
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	tempresults = results[:,0].reshape((paramDiscretization, paramDiscretization))   #np.flipud()
	tempresults = np.transpose(tempresults)
	tempresults_ma = ma.masked_where(tempresults < 0, tempresults)				# Create masked array
	plt.imshow(tempresults_ma, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower', extent=(allPoints[:,0].min()+phiMr[d1], allPoints[:,0].max()+phiMr[d1], allPoints[:,1].min()+phiMr[d2], allPoints[:,1].max()+phiMr[d2]), vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
	if N==3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N==2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if N==3 and topology != "square-open" and topology != "chain":
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
	plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
	plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/imsh_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/imsh_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(4)
	plt.clf()
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	tempresults = results[:,1].reshape((paramDiscretization, paramDiscretization))   #np.flipud()
	tempresults = np.transpose(tempresults)
	tempresults_ma = ma.masked_where(tempresults < 0, tempresults)				# Create masked array
	plt.imshow(tempresults_ma, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower', extent=(allPoints[:,0].min()+phiMr[d1], allPoints[:,0].max()+phiMr[d1], allPoints[:,1].min()+phiMr[d2], allPoints[:,1].max()+phiMr[d2]), vmin=0, vmax=1)
	plt.title(r'last $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
	if N==3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N==2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if N==3 and topology != "square-open" and topology != "chain":
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
	plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
	plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/imsh_PhSpac_lastR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/imsh_PhSpac_lastR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	# plt.figure(5)
	# plt.clf()
	# ax = plt.subplot(1, 1, 1)
	# ax.set_aspect('equal')
	# plt.scatter(allPoints[:,0]+phiMr[d1], allPoints[:,1]+phiMr[d2], c=results[:,0], alpha=0.5, edgecolor='', cmap='jet')#, vmin=0, vmax=1)
	# plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
	# if N==3:
	# 	plt.xlabel(r'$\phi_1^{\prime}$')
	# 	plt.ylabel(r'$\phi_2^{\prime}$')
	# elif N==2:
	# 	plt.xlabel(r'$\phi_0^{\prime}$')
	# 	plt.ylabel(r'$\phi_1^{\prime}$')
	# if N==3 and topology != "square-open" and topology != "chain":
	# 	plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=4)
	# plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
	# plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
	# plt.colorbar()
	# plt.savefig('results/rot_redColor_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	# plt.savefig('results/rot_redColor_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(5)
	plt.clf()
	my_cmap = matplotlib.cm.get_cmap('jet')
	my_cmap.set_under('w')
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	plt.scatter(allPoints[:,0]+phiMr[d1], allPoints[:,1]+phiMr[d2], c=results[:,0], s=10, alpha=0.5, edgecolor='', cmap=my_cmap, vmin=0.0, vmax=1.0)#, vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
	if N==3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N==2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if N==3 and topology != "square-open" and topology != "chain":
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=2)
	plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
	plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/rot_redColor_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/rot_redColor_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure(6)
	plt.clf()
	my_cmap = matplotlib.cm.get_cmap('jet')
	my_cmap.set_under('w')
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')
	plt.scatter(allPoints[:,0]+phiMr[d1], allPoints[:,1]+phiMr[d2], c=results[:,0], s=5, alpha=0.5, edgecolor='', cmap=my_cmap, vmin=0.0, vmax=1.0)#, vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(int(k) ,initPhiPrime0) )
	if N==3:
		plt.xlabel(r'$\phi_1^{\prime}$')
		plt.ylabel(r'$\phi_2^{\prime}$')
	elif N==2:
		plt.xlabel(r'$\phi_0^{\prime}$')
		plt.ylabel(r'$\phi_1^{\prime}$')
	if N==3 and topology != "square-open" and topology != "chain":
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=2)
	plt.xlim([1.05*allPoints[:,0].min()+phiMr[d1], 1.05*allPoints[:,0].max()+phiMr[d1]])
	plt.ylim([1.05*allPoints[:,1].min()+phiMr[d2], 1.05*allPoints[:,1].max()+phiMr[d2]])
	plt.colorbar()
	plt.savefig('results/rot_redColor5_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)
	plt.savefig('results/rot_redColor5_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.draw()
	if show_plot:
		plt.show()

	return 0.0

''' EVALUATION MANY (noisy, dstributed parameters) REALIZATIONS '''
def doEvalManyNoisy(F, Fc, F_Omeg, K, N, Nx, Ny, k, mx, my, delay, c, cPD, Tsim, topology, domega, twistdelta, results, allPoints, dt, orderparam, r, phi, omega_0, K_0, delays_0, show_plot=True):
	orderparam = np.array(orderparam)
	r          = np.array(r)

	if F > 0:																	# for f=0, there would otherwies be a float division by zero
		F1=F
	else:
		F1=1.1

	if (delay == 0):
		t1 = int(round((2./F1)/dt));	t2 = int(round((4./F1)/dt));	t3 = int(round((6./F1)/dt)); t4 = int(round((8./F1)/dt));
		#firstfreq  = (np.diff(phi[:,0:1, :], axis=1)/(dt))						# calculate first frequency of each time series (history of simulation): for each oscillator and realization
		firstfreqs    = np.diff(phi[:,0:4, :], axis=1)/(dt)						# calculate first frequencies of each time series: for each oscillator and realization
		firstfreqsext = np.diff(phi[:,0:t4+4, :], axis=1)/(dt)
		firstfreq     = firstfreqs[:,0,:]
		simstartfreq  = firstfreqs[:,2,:]
		lastfreq      = np.diff(phi[:,-2:, :], axis=1)/(dt)
		lastfreqs     = np.diff(phi[:,-int(2.0*1.0/(F1*dt)):, :], axis=1)/(dt)	# calculate last frequency of each time series: for each oscillator and realization
		lastfreq      = lastfreqs[:,-1:, :]
		#print( 'the results:\n', results, '\n type:', type(results), '\n')
		#print( 'first value in results:\n', results[0], '\n type:', type(results[0]), '\n')
		#print( np.array(results))
	else:
		#firstfreq  = (np.diff(phi[:,0:1, :], axis=1)/(dt))						# calculate first frequency of each time series (history of simulation): for each oscillator and realization
		firstfreqs    = np.diff(phi[:,0:int(round(delay/dt))+4, :], axis=1)/(dt)# calculate first frequencies of each time series: for each oscillator and realization
		firstfreqsext = np.diff(phi[:,0:int(round((8*delay)/dt))+4, :], axis=1)/(dt)
		firstfreq     = firstfreqs[:,0,:]
		simstartfreq  = firstfreqs[:,int(round(delay/dt))+2,:]
		lastfreq      = np.diff(phi[:,-2:, :], axis=1)/(dt)
		lastfreqs     = np.diff(phi[:,-int(2.0*1.0/(F1*dt)):, :], axis=1)/(dt)	# calculate last frequency of each time series: for each oscillator and realization
		lastfreq      = lastfreqs[:,-1:, :]
		#print( 'the results:\n', results, '\n type:', type(results), '\n')
		#print( 'first value in results:\n', results[0], '\n type:', type(results[0]), '\n')
		#print( np.array(results))
	''' SAVE FREQUENCIES '''
	now = datetime.datetime.now()												# provides me the current date and time
	# print('data to be saved: \n', firstfreqsext)
	# print('{shape firstfreqsext, firstfreqsext}: ', firstfreqsext.shape, firstfreqsext)
	# np.savez('results/freqs_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, now.year, now.month, now.day), firstfreqsext=np.array(firstfreqsext), firstfreq=np.array(firstfreq), lastfreq=np.array(lastfreq) )

	'''PLOT TEST'''
	dpi_value  = 300
	histo_bins = 75
	t = np.arange(phi.shape[1])													# plot the phases of the oscillators over time, create "time vector"
	print('\n\nplot data:')
	# print('Uncomment \"matplotlib.use(\'Agg\')\" in ouput.py to enable plotting figures to the desktop, DISABLE for queue-jobs!\n')
	''' HISTOGRAMS PHASES AND FREQUENCIES '''

	if np.std(delays_0.flatten()) > 1E-12:
		plt.figure('histo: static distributed transmission delays')				# plot the distribution of instantaneous frequencies of the history
		plt.clf()
		plt.hist(delays_0.flatten(),
						bins=np.linspace(np.mean(delays_0.flatten())-4.0*np.std(delays_0.flatten()), np.mean(delays_0.flatten())+4.0*np.std(delays_0.flatten()), num=histo_bins),
						rwidth=0.75, histtype='bar', normed=True, log=True)
		plt.title(r'histo: transmission delays $\tau$ of each osci over all realizations', fontdict=titlefont)
		plt.xlabel(r'$\tau$')
		plt.ylabel(r'log[$P(\tau)$]')
		plt.savefig('results/hist_transdelays_static_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/hist_transdelays_static_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	if np.std(K_0.flatten()) > 1E-12:
		plt.figure('histo: static distributed coupling strength')				# plot the distribution of instantaneous frequencies of the history
		plt.clf()
		plt.hist(K_0.flatten(),
						bins=np.linspace(np.mean(K_0.flatten())-4.0*np.std(K_0.flatten()), np.mean(K_0.flatten())+4.0*np.std(K_0.flatten()), num=histo_bins),
						rwidth=0.75, histtype='bar', normed=True, log=True)
		plt.title(r'histo: coupling strengths $K$ of each osci over all realizations', fontdict=titlefont)
		plt.xlabel(r'$K$')
		plt.ylabel(r'log[$P(K)$]')
		plt.savefig('results/hist_distCoupStr_static_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/hist_distCoupStr_static_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	if np.std(omega_0.flatten()) > 1E-12:
		plt.figure('histo: intrinsic frequencies of oscillators drawn at setup: omega_k(-delay)')
		plt.clf()
		plt.hist(omega_0.flatten(),
						bins=np.linspace(np.mean(omega_0.flatten())-4.0*np.std(omega_0.flatten()), np.mean(omega_0.flatten())+4.0*np.std(omega_0.flatten()), num=histo_bins),
						rwidth=0.75, histtype='bar', normed=True, log=True)
		plt.title(r'mean intrinsic frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( np.mean(omega_0.flatten())/(2.0*np.pi), np.std(omega_0.flatten())/(2.0*np.pi) ) , fontdict=titlefont)
		plt.xlabel(r'frequency bins [rad/s]')
		plt.ylabel(r'$log[g(\omega)]$')
		plt.savefig('results/omega0_intfreq_histo_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/omega0_intfreq_histo_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	# if F > 0:
	# 	SetLog=str(True)
	# else:
	# 	SetLog=str(False)
	if c > 0 or cPD > 0:
		plt.figure('histo: orderparameter at t=TSim over all realizations') 		# plot the distribution of order parameters
		plt.clf()
		binrange = 4*np.std(orderparam[:,-1])
		plt.hist(orderparam[:,-1],
						bins=np.linspace(np.mean(orderparam[:,-1])-binrange, np.mean(orderparam[:,-1])+binrange, num=histo_bins),
						rwidth=0.75, histtype='bar', normed=True, log=True)
		plt.title(r'histo: orderparam $R(t_{end})$ of each osci over all realizations', fontdict=titlefont)
		plt.xlabel(r'R($t_{end}$)')
		plt.ylabel(r'loghist[$R(t_{end})$]')
		plt.savefig('results/hist_orderparam_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/hist_orderparam_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)
	else:
		print('No output of histograms since there is no noise in the system, hence deterministic!')

	if c > 0 or cPD > 0:
		plt.figure('histo: phases at t={0, TSim} over all realizations and oscis - shifted dist. at TSim to mean of first')	# plot the distribution of instantaneous frequencies of the history
		plt.clf()
		shift_mean0 = ( np.mean( phi[:,-1,:].flatten() ) - np.mean( phi[:,int(round(delay/dt))+1,:].flatten() ) ) # distance between the mean values of the distribution at different times
		common_bins=np.linspace(min( np.array([(phi[:,int(round(delay/dt))+1,:].flatten()+shift_mean0), (phi[:,-1,:].flatten())]).flatten() ),
								max( np.array([(phi[:,int(round(delay/dt))+1,:].flatten()+shift_mean0), (phi[:,-1,:].flatten())]).flatten() ),
								num=histo_bins)
		# print(phi[:,int(round(delay/dt))+1,:].flatten() + shift_mean)
		plt.hist(phi[:,int(round(delay/dt))+1,:].flatten()+shift_mean0,
				bins=common_bins,
				color='b', rwidth=0.75, histtype='bar', label='t=0', normed=True, log=True)
		plt.hist( ( phi[:,-1,:].flatten() ),
				bins=common_bins,
				color='r', rwidth=0.75, histtype='bar', label='t=TSim', normed=True, log=True)
		#plt.xlim((1.1*min(firstfreq), 1.1*max(firstfreq)))
		plt.legend(loc='best')
		plt.title(r't=0: mean phase $\bar{\phi}=%.3f$ and std $\sigma_{\phi}=%.5f$  [Hz]' %( np.mean(phi[:,int(round(delay/dt))+1,:].flatten()), np.mean(np.std(phi[:,int(round(delay/dt))+1,:], axis=1),axis=0) ) , fontdict=titlefont)
		plt.xlabel(r'$\phi(t=0,TSim)$')
		plt.ylabel(r'loghist$[\phi(t=0,TSim)]$')
		plt.savefig('results/histo_phases_t0_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/histo_phases_t0_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	if (delay == 0):
		plt.figure('histo: phases over all oscis at t={2,4,6,8}*Tf shifted to same mean')	# plot a histogram of the last frequency of each oscillators over all realizations
		plt.clf()
		# print('mean phase at t = 8*T_w', np.mean( phi[:,int(round(8*delay/dt)),:].flatten() ) )
		t1 = int(round((2./F1)/dt)); label1 = 't=2*Tf'
		t2 = int(round((4./F1)/dt)); label2 = 't=4*Tf'
		t3 = int(round((6./F1)/dt)); label3 = 't=6*Tf'
		t4 = int(round((8./F1)/dt)); label4 = 't=8*Tf'
		plt.title(r't=$8$Tf: mean phase $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(phi[:,(t4-10),:].flatten())/(2.0*np.pi), np.mean(np.std(phi[:,(t4-10),:], axis=1), axis=0)/(2.0*np.pi) ), fontdict=titlefont)
	else:
		plt.figure('histo: phases over all oscis at t={2,4,6,8}*delay shifted to same mean')# plot a histogram of the last frequency of each oscillators over all realizations
		plt.clf()
		# print('mean phase at t = 8*tau', np.mean( phi[:,int(round(8*delay/dt)),:].flatten() ) )
		t1 = int(round((2.*delay)/dt)); label1 = 't=2*tau'
		t2 = int(round((4.*delay)/dt)); label2 = 't=4*tau'
		t3 = int(round((6.*delay)/dt)); label3 = 't=6*tau'
		t4 = int(round((8.*delay)/dt)); label4 = 't=8*tau'
		plt.title(r't=$8\tau$: mean phase $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(phi[:,(t4-10),:].flatten())/(2.0*np.pi), np.mean(np.std(phi[:,(t4-10),:], axis=1), axis=0)/(2.0*np.pi) ), fontdict=titlefont)
	common_bins=np.linspace( ( np.mean( phi[:,t4,:].flatten() ) - 4.0*np.std( phi[:,t4,:].flatten() ) ),
							 ( np.mean( phi[:,t4,:].flatten() ) + 4.0*np.std( phi[:,t4,:].flatten() ) ),
							 num=histo_bins)
	shift_mean1 = np.mean( phi[:,t4,:].flatten() ) - np.mean( phi[:,t1,:].flatten() )
	shift_mean2 = np.mean( phi[:,t4,:].flatten() ) - np.mean( phi[:,t2,:].flatten() )
	shift_mean3 = np.mean( phi[:,t4,:].flatten() ) - np.mean( phi[:,t3,:].flatten() )
	plt.hist( ( phi[:,t1,:].flatten() + shift_mean1),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.9, label=label1 )
	plt.hist( ( phi[:,int(round((4.*delay)/dt)),:].flatten() + shift_mean2),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label=label2 )
	plt.hist( ( phi[:,int(round((6.*delay)/dt)),:].flatten() + shift_mean3),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.6, label=label3 )
	plt.hist( phi[:,int(round((8.*delay)/dt)-10),:].flatten(),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.45, label=label4 )
	plt.legend()
	plt.xlabel(r'phase bins [rad]')
	plt.ylabel(r'$\log[hist(\phi)]$')
	plt.savefig('results/histo_phases_all_osci_diff_times_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/histo_phases_all_osci_diff_times_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	if c > 0 or cPD > 0:
		plt.figure('histo: last instant. freq. over all oscis and realizations at t=TSim')			   # plot a histogram of the last frequency of each oscillators over all realizations
		plt.clf()
		plt.hist(lastfreq[:,:,:].flatten(),
							bins=np.linspace( ( np.mean(lastfreq[:,:,:].flatten())-4.0*np.std(lastfreq[:,:,:].flatten()) ), ( np.mean(lastfreq[:,:,:].flatten())+4.0*np.std(lastfreq[:,:,:].flatten()) ),
							num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar')  #, color=(1, np.random.rand(1), np.random.rand(1)) )
		plt.title(r't=TSim: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( np.mean(lastfreq[:,0,:].flatten())/(2.0*np.pi), np.mean(np.std(lastfreq[:,0,:], axis=1),axis=0)/(2.0*np.pi) ), fontdict=titlefont)
		plt.xlabel(r'frequency bins [rad/s]')
		plt.ylabel(r'loghist$[\dot{\phi}(t=TSim)]$')
		plt.savefig('results/histo_lastfreq_all_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
		plt.savefig('results/histo_lastfreq_all_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

		plt.figure('histo: instant. freq. over all oscis and realizations at t={delay+0.1Tf, TSim}')   # plot a histogram of the last frequency of each oscillators over all realizations
		plt.clf()
		if (delay==0):
			common_bins = np.linspace(	min( np.array([firstfreqsext[:,int(round((.1/F1)/dt)),:].flatten(), lastfreq[:,:,:].flatten()]).flatten() ),
								  	max( np.array([firstfreqsext[:,int(round((.1/F1)/dt)),:].flatten(), lastfreq[:,:,:].flatten()]).flatten() ),
									num=histo_bins )
			plt.hist(firstfreqsext[:,int(round((.1/F1)/dt)),:].flatten(), bins=common_bins,
								rwidth=0.75, normed=True, log=True, histtype='bar', label='t=0.1*Tf')
		else:
			common_bins = np.linspace(	min( np.array([firstfreqsext[:,int(round((.1/F1)/dt)),:].flatten(), lastfreq[:,:,:].flatten()]).flatten() ),
								  	max( np.array([firstfreqsext[:,int(round((.1/F1)/dt)),:].flatten(), lastfreq[:,:,:].flatten()]).flatten() ),
									num=histo_bins )
			plt.hist(firstfreqsext[:,int(round((.1/F1)/dt)),:].flatten(), bins=common_bins,
							rwidth=0.75, normed=True, log=True, histtype='bar', label='t=0.1*Tf')
		plt.hist(lastfreq[:,:,:].flatten(), bins=common_bins,
							rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=TSim')
		plt.legend()
		plt.title(r'at t=$%.3f$: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( (delay+.1*(1/F1)),
																							  np.mean(firstfreqsext[:,int(round((delay+.1*(1/F1))/dt)),:].flatten()/(2.0*np.pi)),
																							  np.mean(np.std(firstfreqsext[:,int(round((delay+.1*(1/F1))/dt)),:], axis=1), axis=0)/(2.0*np.pi) ), fontdict=titlefont)
		plt.xlabel(r'frequency bins [rad/s]')
		plt.ylabel(r'loghist[$\dot{\phi}$]')
		plt.savefig('results/histoFreqAllOsci_2timePoints_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.month, now.day))
		plt.savefig('results/histoFreqAllOsci_2timePoints_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.month, now.day), dpi=dpi_value)


	if (delay == 0):
		plt.figure('histo: instant. freq. over all oscis at t={2,4,6,8}*Tf')		# plot a histogram of the last frequency of each oscillators over all realizations
		plt.clf()
		t1 = int(round((2./F1)/dt)); label1 = 't=2*Tf'
		t2 = int(round((4./F1)/dt)); label2 = 't=4*Tf'
		t3 = int(round((6./F1)/dt)); label3 = 't=6*Tf'
		t4 = int(round((8./F1)/dt)); label4 = 't=8*Tf'
		common_bins = np.linspace( 	min( np.array([firstfreqsext[:,t1,:].flatten(), firstfreqsext[:,t4,:].flatten()]).flatten() ),
									max( np.array([firstfreqsext[:,t1,:].flatten(), firstfreqsext[:,t4,:].flatten()]).flatten() ),
									num=histo_bins )
		plt.hist(firstfreqsext[:,t1,:].flatten(), bins=common_bins,
		 					rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.9, label='t=2*Tf')
		plt.hist(firstfreqsext[:,t2,:].flatten(), bins=common_bins,
							rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=4*Tf')
		plt.hist(firstfreqsext[:,t3,:].flatten(), bins=common_bins,
							rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.6, label='t=6*Tf')
		plt.hist(firstfreqsext[:,t4,:].flatten(), bins=common_bins,
							rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.45, label='t=8*Tf')
		plt.title(r't=$8$Tf: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( np.mean(firstfreqsext[:,(t4-10),:].flatten())/(2.0*np.pi), np.mean(np.std(firstfreqsext[:,(t4-10),:], axis=1), axis=0)/(2.0*np.pi) ), fontdict=titlefont)
	else:
		plt.figure('histo: instant. freq. over all oscis at t={2,4,6,8}*delay')		# plot a histogram of the last frequency of each oscillators over all realizations
		plt.clf()
		t1 = int(round((2.*delay)/dt)); label1 = 't=2*tau'
		t2 = int(round((4.*delay)/dt)); label2 = 't=4*tau'
		t3 = int(round((6.*delay)/dt)); label3 = 't=6*tau'
		t4 = int(round((8.*delay)/dt)); label4 = 't=8*tau'
		common_bins = np.linspace( 	min( np.array([firstfreqsext[:,t1,:].flatten(), firstfreqsext[:,t4,:].flatten()]).flatten() ),
									max( np.array([firstfreqsext[:,t1,:].flatten(), firstfreqsext[:,t4,:].flatten()]).flatten() ),
									num=histo_bins )
		plt.hist(firstfreqsext[:,t1,:].flatten(), bins=common_bins,
		 					rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.9, label='t=2*tau')
		plt.hist(firstfreqsext[:,t2,:].flatten(), bins=common_bins,
							rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=4*tau')
		plt.hist(firstfreqsext[:,t3,:].flatten(), bins=common_bins,
							rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.6, label='t=6*tau')
		plt.hist(firstfreqsext[:,t4,:].flatten(), bins=common_bins,
							rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.45, label='t=8*tau')
		plt.title(r't=$8\tau$: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( np.mean(firstfreqsext[:,(t4-10),:].flatten())/(2.0*np.pi), np.mean(np.std(firstfreqsext[:,(t4-10),:], axis=1), axis=0)/(2.0*np.pi) ), fontdict=titlefont)

	plt.legend()
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'loghist$[\dot{\phi}]$')
	plt.savefig('results/histo_freq_all_osci_diff_times_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/histo_freq_all_osci_diff_times_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	''' TIME EVOLUTION '''
	############################################################################

	plt.figure('order parameter time series for each realization and mean')
	plt.clf()
	for i in range (orderparam.shape[0]):
		# print('Plot order parameter of realization #', i)
		plt.plot(t*dt,orderparam[i,:], alpha=0.2)
	plt.plot(t*dt,np.mean(orderparam[:,:], axis=0), label='mean order param', color='r')
	plt.plot(delay, orderparam[0,int(round(delay/dt))], 'yo', ms=5)# mark where the simulation starts
	plt.title(r'mean order parameter, with $1-\bar{R}_{\rm Nsim}(TSim)=%.3e$, and $\sigma_{\bar{R}_{\rm Nsim}(TSim)}=%.3e$' %(1-np.mean(orderparam[-1,:]), np.std(orderparam[-1,:])), fontdict=titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict=labelfont)
	plt.ylabel(r'$R( t,m = %d )$' % k, fontdict=labelfont)
	plt.legend(loc='center right')
	plt.savefig('results/orderParams-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/orderParams-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure('time evolution of mean and standard deviation of phases for first and all realization')
	plt.clf()
	plt.subplot(2,1,1)
	plt.title('time evolution of mean and std of phases for first and all realizations', fontdict=titlefont)
	plt.plot(t*dt, np.mean(phi[0,:,:], axis=1), label=r'$\langle \phi_{first} \rangle$')
	plt.plot(t*dt, np.mean(phi[:,:,:], axis=tuple(range(0, 3, 2))), label=r'$\langle \phi_{all} \rangle$')
	plt.ylabel(r'$\bar{\phi}$ [rad]', fontdict = labelfont)
	plt.legend(loc='upper left')
	plt.subplot(2,1,2)
	plt.plot(t*dt, np.std(phi[0,:,:], axis=1), label=r'$\sigma_{\phi,first}$')
	plt.plot(t*dt, np.mean(np.std(phi[:,:,:], axis=2), axis=0), label=r'$\sigma_{\phi,all}$')
	plt.legend(loc='upper left')
	plt.xlabel(r'$t$ $[s]$', fontdict=labelfont)
	plt.ylabel(r'$\sigma_{\phi}$ [rad]', fontdict=labelfont)
	plt.savefig('results/mean_of_phase_vs_t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/mean_of_phase_vs_t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure('time evolution of mean and standard deviation of frequencies for first and all realizations')
	plt.clf()
	plt.subplot(2,1,1)
	plt.title('time evolution of mean and std of frequencies for first and all realizations', fontdict=titlefont)
	plt.plot(t[0:-1]*dt, np.mean(np.diff(phi[0,:,:], axis=0)/dt, axis=1), label=r'$\langle \dot{\phi}_{first} \rangle$')
	plt.plot(t[0:-1]*dt, np.mean(np.diff(phi[:,:,:], axis=1)/dt, axis=tuple(range(0, 3, 2))), label=r'$\langle \dot{\phi}_{all} \rangle$')
	plt.legend(loc='best')
	plt.ylabel(r'$\dot{\bar{\phi}}$ [rad Hz]')
	plt.subplot(2,1,2)
	plt.plot(t[0:-1]*dt, np.std(np.diff(phi[0,:,:], axis=0)/dt, axis=1), label=r'$\sigma_{\dot{\phi},first}$')
	plt.plot(t[0:-1]*dt, np.std(np.diff(phi[:,:,:], axis=1)/dt, axis=tuple(range(0, 3, 2))), label=r'$\sigma_{\dot{\phi},all}$')
	plt.legend(loc='best')
	plt.xlabel(r'$t$ $[s]$', fontdict=labelfont)
	plt.ylabel(r'$\sigma_{\dot{\phi}}$ [rad Hz]', fontdict=labelfont)
	plt.savefig('results/mean_of_freq_vs_t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/mean_of_freq_vs_t_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure('plot phase time-series for first realization')					# plot the phase time-series of all oscis and all realizations
	plt.clf()
	for i in range (phi.shape[2]):
		#print('Plot now osci #', i, '\n')
		plt.plot((t*dt),phi[0,:,i])
	# print(np.arange(0,len(phi),2*np.pi*F_Omeg*dt))
	# plt.plot(phi[:,1]-2*np.pi*dt*F_Omeg*np.arange(len(phi)))					# plots the difference between the expected phase (according to Omega) vs the actual phase evolution
	plt.plot(delay, phi[0,int(round(delay/dt)),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'time series phases, inst. freq: $\dot{\phi}_0(t_{start})=$%.4f, $\dot{\phi}_0(t_{end})=$%.4f [rad Hz]' %( ((phi[0][11][0]-phi[0][1][0])/(10*dt)), ((phi[0][-4][0]-phi[0][-14][0])/(10*dt)) ), fontdict=titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict=labelfont)
	plt.ylabel(r'$\phi(t)$', fontdict=labelfont)
	plt.savefig('results/phases-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/phases-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure('plot instantaneous-frequencies for first realization')			# plot the frequencies of the oscillators over time
	plt.clf()
	for i in range (phi.shape[2]):												# for each oscillator: plot of frequency vs time
		plt.plot((t[0:-1]*dt), np.transpose( np.diff(phi[0,:,i])/dt ))
	phidot = np.diff(phi[:,:,0], axis=1)/dt
	plt.plot(delay-dt, phidot[0,int(round(delay/dt)-1)], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F1*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	# print(t[-int(2*1.0/(F1*dt))], t[-1])
	plt.title(r'mean frequency over last $2T$-eigenperiods $\bar{\omega}=$%.4f [rad Hz]' % np.mean(phidot[-int(round(2*1.0/(F1*dt))):, 0] ), fontdict=titlefont)
	plt.xlabel(r't [s]', fontdict = labelfont)
	plt.ylabel(r'$\dot{\phi}(t)$ [rad Hz]', fontdict=labelfont)
	plt.savefig('results/freq-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day))
	plt.savefig('results/freq-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.4f_c%.7e_cPD%.7e_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day), dpi=dpi_value)

	plt.draw()
	if show_plot:
		plt.show()

	if len(orderparam[:,0]) > 24:												# only save if many realizations have been computed
		# SAVE DATA, APPEND TO RESULTS FILE
		print('Writing results to noisy_results.csv')
		temp = []; TempOrdParame = [];
		with open('results/1Anoisy_res.csv', 'a') as f:							# 'a' means append to file! other modes: 'w' write only and replace existing file, 'r' readonly...
			writer = csv.writer(f, delimiter=',') 								#, dtype={'K':np.float, 'Fc':np.float, 'delay':np.float, 'F_Omeg':np.float, 'k':np.int,
																				#'Tsim':np.int, 'id':np.int, 'ReLambda':np.float, 'SimSeconds':np.float, 'topology':np.str,
																				#'c':np.float, 'N':np.int, 'couplingfct':np.str, 'Nx':np.int, 'Ny':np.int, 'mx':np.int, 'my':np.int})
			#lastLineCsvF = len(param_cases_csv.sort_values('id'))+3			# add 3 in order to account for the header of the csv file
			mlastFreqReal = np.mean(lastfreqs)									# the mean of all last frequencies
			mstdFreqReali = np.std(lastfreqs)									# the std of all last frequencies
			mlastOrderPar = np.mean(orderparam[-1,:])							# the mean over the last orderparameter value of each realization
			mstdOrderPara = np.std(orderparam[-1,:])							# the std over the last orderparameter value of each realization
			TempOrdParame = orderparam[:,-int(round(2*1.0/(F1*dt))):]			# extract for all realizations the order parameter over 2 periods of omega at the end
			Realizations  = len(orderparam[:,0])								# determine the number of realizations that have been computed
			#print('extraction order parameter:', TempOrdParame)
			TempOrdParame = np.mean(TempOrdParame, axis=1)
			print('extraction order parameter, mean over 2T:', TempOrdParame)
			mAverOrderPar = np.mean(TempOrdParame)								# the mean of all last orderparameter values, each order parameter averaged over 2T at the end
			mstdAverOrdPa = np.std(TempOrdParame)								# the std of all last orderparameter values, each order parameter averaged over 2T at the end
			# doEvalManyNoisy(F, Fc, F_Omeg, K, N, Nx, Ny, k, mx, my, delay, c, cPD, domega, twistdelta, results, allPoints, dt, orderparam, r, phi, omega_0, K_0, delays_0, show_plot=True):
			# write row Nx, Ny, mx, my, K, Fc, delay, topology, F_Omeg, Tsim, ReLambda, c, cPD, orderparam

			temp = [str(int(Nx)), str(int(Ny)), str(int(mx)), str(int(my)), str(float(K)), str(float(Fc)), str(float(delay)), str(topology), str(float(F1)),
					str(float(F_Omeg)), str(int(Tsim)), str(int(Realizations)), str(float(c)), str(float(cPD)), str(float(mlastOrderPar)), str(float(mstdOrderPara)),
					str(float(mAverOrderPar)), str(float(mstdAverOrdPa)), str(float(mlastFreqReal)), str(float(mstdFreqReali))]	#list of strings containing the information
			print('\nWRITEOUT:', temp, '\n')
			writer.writerow(temp)
