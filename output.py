#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import evaluation as eva
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import datetime

''' All plots in latex mode '''
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

''' STYLEPACKS '''
titlefont = {
        'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 20,
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
def plotTimeSeries(phi, F, Fc, dt, orderparam, k, delay, F_Omeg, K, coupFct, Tsim, Fsim=None):

	phi = phi[:,:,:]; orderparam = orderparam[0,:]								# there is only one realization of interest -reduces dimensionof phi array
	afterTransients = int( round( 0.5*Tsim / dt ) )
	phiSpect = phi[:,-afterTransients:,:]
	if coupFct == 'triang':
		print('Calculate spectrum for square wave signals. Fsim=%d' %Fsim)
		f, Pxx_db = eva.calcSpectrum( (phiSpect), Fsim, 'square')				# calculate spectrum of signals, i.e., of this state
	elif coupFct == 'sin':
		print('check that... sine coupFct only if cos and sin signal input')
		f, Pxx_db = eva.calcSpectrum( (phiSpect), Fsim, 'sin')					# calculate spectrum of signals, i.e., of this state
	elif coupFct == 'cos':
		print('Calculate spectrum for cosinusoidal signals. Fsim=%d' %Fsim)
		f, Pxx_db = eva.calcSpectrum( (phiSpect), Fsim, 'sin')					# calculate spectrum of signals, i.e., of this state

	now = datetime.datetime.now()

	plt.figure('spectrum of synchronized state')								# plot spectrum
	plt.clf()
	for i in range (len(f)):
		plt.plot(f[i], Pxx_db[i], '-')
	plt.title('power spectrum', fontdict = titlefont)
	plt.xlim(0,F+20*K);	#plt.ylim(-100,0);
	plt.xlabel('frequencies [Hz]', fontdict = labelfont); plt.ylabel('P [dB]', fontdict = labelfont)
	plt.grid()
	plt.savefig('results/powerdensity_dB_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/powerdensity_dB_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=300)

	phi = phi[0,:,:];															# from here on the phi array is reduced in dimension - realization 0 picked
	t = np.arange(phi.shape[0])													# plot the phases of the oscillators over time

	plt.figure('histogram of frequencies')										# plot a histogram of the frequencies of the oscillators over time
	plt.clf()
	lastfreqs = (np.diff(phi[-int(2*1.0/(F*dt)):, :], axis=0).flatten()/(dt))
	plt.hist(lastfreqs, bins=np.linspace(2*np.pi*(F-K), 2*np.pi*(F+K), num=21), rwidth=0.75 )
	plt.axvspan(t[-int(2*1.0/(F*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.xlim((2*np.pi*(F-K), 2*np.pi*(F+K)))
	plt.title(r'mean frequency [Hz] $\bar{f}=$%.3f and std $\bar{\sigma}_f=$%.4f' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/(2.0*np.pi) ), fontdict = titlefont)
	plt.xlabel(r'$\dot{\phi}(-2T -> T_{end})$ $[rad/s]$', fontdict = labelfont)
	plt.ylabel(r'histogram', fontdict = labelfont)
	plt.savefig('results/freq_histo_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/freq_histo_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=300)

	plt.figure('phases over time')
	plt.clf()
	plt.plot((t*dt),phi)
	plt.plot(delay, phi[int(round(delay/dt)),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'time series phases, $\dot{\phi}_0(t_{start})=%.4f$, $\dot{\phi}_0(t_{end})=%.4f$  [rad/Hz]' %( ((phi[2][0]-phi[1][0])/(dt)), ((phi[-4][0]-phi[-5][0])/(dt)) ), fontdict = titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
	plt.ylabel(r'$\phi(t)$', fontdict = labelfont)
	plt.savefig('results/phases-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/phases-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=300)
	print(r'frequency of zeroth osci at the beginning and end of the simulation:, freqStart=%.4f, freqEnd=%.4f  [rad/Hz]' %(((phi[2][0]-phi[1][0])/(dt)), ((phi[-4][0]-phi[-5][0])/(dt))) )
	print('last values of the phases:\n', phi[-3:,:])

	plt.figure('frequencies over time')											# plot the frequencies of the oscillators over time
	plt.clf()
	phidot = np.diff(phi, axis=0)/dt
	plt.plot((t[0:-1]*dt),phidot)
	plt.plot(delay-dt, phidot[int(round(delay/dt)-1),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'mean frequency [rad Hz] of last $2T$-eigenperiods $\dot{\bar{\phi}}=$%.4f' % np.mean(phidot[-int(round(2*1.0/(F*dt))):, 0] ), fontdict = titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
	plt.ylabel(r'$\dot{\phi}(t)$ $[rad Hz]$', fontdict = labelfont)
	plt.savefig('results/freq-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/freq-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=300)

	plt.figure('order parameter over time')										# plot the order parameter in dependence of time
	plt.clf()
	plt.plot((t*dt), orderparam)
	plt.plot(delay, orderparam[int(round(delay/dt))], 'yo', ms=5)				# mark where the simulation starts
	plt.axvspan(t[-int(2*1.0/(F*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'mean order parameter $\bar{R}=$%.2f, and $\bar{\sigma}=$%.4f' %(np.mean(orderparam[-int(round(2*1.0/(F*dt))):]), np.std(orderparam[-int(round(2*1.0/(F*dt))):])), fontdict = titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict = labelfont)
	plt.ylabel(r'$R( t,m = %d )$' % k, fontdict = labelfont)
	plt.savefig('results/orderP-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/orderP-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=300)
	#print('\nlast entry order parameter: R-1 = %.3e' % (orderparam[-1]-1) )
	#print('\nlast entries order parameter: R = ', orderparam[-25:])

	# plt.draw()

''' EVALUATION BRUTE-FORCE BASIN OF ATTRACTION '''
def doEvalBruteForce(Fc, F_Omeg, K, N, k, delay, twistdelta, results, allPoints, initPhiPrime0, paramDiscretization, delays_0):
	''' Here addtional output, e.g., graphs or matrices can be implemented for testing '''
	# we want to plot all the m-twist locations in rotated phase space: calculate phases, rotate and then plot into the results
	twist_points  = np.zeros((N, N), dtype=np.float)							# twist points in physical phase space
	twist_pointsR = np.zeros((N, N), dtype=np.float)							# twist points in rotated phase space
	alltwistP = []
	now = datetime.datetime.now()

	if N == 2:																	# this part is for calculating the points of m-twist solutions in the rotated space, they are plotted later
		pass
	if N == 3:
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
		alltwistPR= np.transpose(eva.rotate_phases(np.transpose(alltwistP), isInverse=True))	# rotate the points into rotated phase space
		# print('alltwistP rotated (alltwistPR):\n', alltwistPR, '\n')

	''' PLOTS '''
	plt.figure(1)																#
	plt.clf()
	plt.scatter(allPoints[:,0], allPoints[:,1], c=results[:,0], alpha=0.5, edgecolor='')
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(k ,initPhiPrime0) )
	plt.xlabel(r'$\phi_1^{\prime}$')
	plt.ylabel(r'$\phi_2^{\prime}$')
	plt.xlim([1.05*allPoints[:,0].min(), 1.05*allPoints[:,0].max()])
	plt.ylim([1.05*allPoints[:,1].min(), 1.05*allPoints[:,1].max()])
	plt.colorbar()
	if N==3:
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
	plt.savefig('results/rot_red_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/rot_red_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=300)

	plt.figure(2)
	plt.clf()
	plt.scatter(allPoints[:,0], allPoints[:,1], c=results[:,1], alpha=0.5, edgecolor='')
	plt.title(r'last $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(k ,initPhiPrime0) )
	plt.xlabel(r'$\phi_1^{\prime}$')
	plt.ylabel(r'$\phi_2^{\prime}$')
	plt.xlim([1.05*allPoints[:,0].min(), 1.05*allPoints[:,0].max()])
	plt.ylim([1.05*allPoints[:,1].min(), 1.05*allPoints[:,1].max()])
	plt.colorbar()
	if N==3:
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
	plt.savefig('results/rot_red_PhSpac_lastR_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/rot_red_PhSpac_lastR_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=300)

	plt.figure(3)
	plt.clf()
	tempresults = results[:,0].reshape((paramDiscretization, paramDiscretization))   #np.flipud()
	tempresults = np.transpose(tempresults)
	plt.imshow(tempresults, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower', extent=(allPoints[:,0].min(), allPoints[:,0].max(), allPoints[:,1].min(), allPoints[:,1].max()), vmin=0, vmax=1)
	plt.title(r'mean $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(k ,initPhiPrime0) )
	plt.xlabel(r'$\phi_1^{\prime}$')
	plt.ylabel(r'$\phi_2^{\prime}$')
	plt.colorbar()
	if N==3:
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
		plt.xlim([1.05*allPoints[:,0].min(), 1.05*allPoints[:,0].max()])
		plt.ylim([1.05*allPoints[:,1].min(), 1.05*allPoints[:,1].max()])
	plt.savefig('results/imsh_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/imsh_PhSpac_meanR_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=1000)

	plt.figure(4)
	plt.clf()
	tempresults = results[:,1].reshape((paramDiscretization, paramDiscretization))   #np.flipud()
	tempresults = np.transpose(tempresults)
	plt.imshow(tempresults, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower', extent=(allPoints[:,0].min(), allPoints[:,0].max(), allPoints[:,1].min(), allPoints[:,1].max()), vmin=0, vmax=1)
	plt.title(r'last $R(t,m)$, constant dim: $\phi_0^{\prime}=%.2f$' % initPhiPrime0)
	plt.title(r'last $R(t,m=%d )$, constant dim: $\phi_0^{\prime}=%.2f$' %(k ,initPhiPrime0) )
	plt.xlabel(r'$\phi_1^{\prime}$')
	plt.ylabel(r'$\phi_2^{\prime}$')
	plt.colorbar()
	if N==3:
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
		plt.xlim([1.05*allPoints[:,0].min(), 1.05*allPoints[:,0].max()])
		plt.ylim([1.05*allPoints[:,1].min(), 1.05*allPoints[:,1].max()])
	plt.savefig('results/imsh_PhSpac_lastR_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/imsh_PhSpac_lastR_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=1000)

	plt.draw()
	plt.show()

	return 0.0

''' EVALUATION MANY (noisy, dstributed parameters) REALIZATIONS '''
def doEvalManyNoisy(F, Fc, F_Omeg, K, N, k, delay, domega, twistdelta, results, allPoints, dt, orderparam, r, phi, omega_0, K_0, delays_0):
	orderparam = np.array(orderparam)
	r          = np.array(r)

	#firstfreq  = (np.diff(phi[:,0:1, :], axis=1)/(dt))						# calculate first frequency of each time series (history of simulation): for each oscillator and realization
	firstfreqs    = (np.diff(phi[:,0:int(round(delay/dt))+4, :], axis=1)/(dt))	# calculate first frequencies of each time series: for each oscillator and realization
	firstfreqsext = (np.diff(phi[:,0:int(round((8*delay)/dt))+4, :], axis=1)/(dt))
	firstfreq     = firstfreqs[:,0,:]
	simstartfreq  = firstfreqs[:,int(round(delay/dt))+2,:]
	lastfreq      = (np.diff(phi[:,-2:, :], axis=1)/(dt))
	lastfreqs     = (np.diff(phi[:,-int(2.0*1.0/(F*dt)):, :], axis=1)/(dt))	# calculate last frequency of each time series: for each oscillator and realization
	lastfreq      = lastfreqs[:,-1:, :]
	#print( 'the results:\n', results, '\n type:', type(results), '\n')
	#print( 'first value in results:\n', results[0], '\n type:', type(results[0]), '\n')
	#print( np.array(results))
	''' SAVE FREQUENCIES '''
	now = datetime.datetime.now()												# provides me the current date and time
	np.savez('results/freqs_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), data=[firstfreqsext, firstfreq, lastfreq])

	'''PLOT TEST'''
	dpi_value  = 300
	histo_bins = 75
	t = np.arange(phi.shape[1])													# plot the phases of the oscillators over time, create "time vector"

	print('plot data:\n')
	''' HISTOGRAMS PHASES AND FREQUENCIES '''

	if np.std(delays_0.flatten()) > 1E-15:
		plt.figure('histo: static distributed transmission delays')				 # plot the distribution of instantaneous frequencies of the history
		plt.clf()
		plt.hist(delays_0.flatten(),
						bins=np.linspace(np.mean(delays_0.flatten())-4.0*np.std(delays_0.flatten()), np.mean(delays_0.flatten())+4.0*np.std(delays_0.flatten()), num=histo_bins),
						rwidth=0.75, histtype='bar', normed=True, log=True)
		plt.title(r'histo: transmission delays $\tau$ of each osci over all realizations')
		plt.xlabel(r'$\tau$')
		plt.ylabel(r'log[$P(\tau)$]')
		plt.savefig('results/hist_transdelays_static_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	if np.std(K_0.flatten()) > 1E-15:
		plt.figure('histo: static distributed coupling strength')				# plot the distribution of instantaneous frequencies of the history
		plt.clf()
		plt.hist(K_0.flatten(),
						bins=np.linspace(np.mean(K_0.flatten())-4.0*np.std(K_0.flatten()), np.mean(K_0.flatten())+4.0*np.std(K_0.flatten()), num=histo_bins),
						rwidth=0.75, histtype='bar', normed=True, log=True)
		plt.title(r'histo: coupling strengths $K$ of each osci over all realizations')
		plt.xlabel(r'$K$')
		plt.ylabel(r'log[$P(K)$]')
		plt.savefig('results/hist_distCoupStr_static_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	if np.std(omega_0.flatten()) > 1E-15:
		plt.figure('histo: intrinsic frequencies of oscillators drawn at setup: omega_k(-delay)')
		plt.clf()
		plt.hist(omega_0.flatten(),
						bins=np.linspace(np.mean(omega_0.flatten())-4.0*np.std(omega_0.flatten()), np.mean(omega_0.flatten())+4.0*np.std(omega_0.flatten()), num=histo_bins),
						rwidth=0.75, histtype='bar', normed=True, log=True)
		plt.title(r'mean intrinsic frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( np.mean(omega_0.flatten())/(2.0*np.pi), np.std(omega_0.flatten())/(2.0*np.pi) ) )
		plt.xlabel(r'frequency bins [rad/s]')
		plt.ylabel(r'$log[g(\omega)]$')
		plt.savefig('results/omega0_intfreq_histo_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	plt.figure('histo: orderparameter at t=TSim over all realizations') 		# plot the distribution of instantaneous frequencies of the history
	plt.clf()
	for i in range (phi.shape[0]):												# loop over all realizations
		plt.hist(orderparam[:,-1],
						bins=np.linspace(np.mean(orderparam[:,-1])-4*np.std(orderparam[:,-1]), np.mean(orderparam[:,-1])+4*np.std(orderparam[:,-1]), num=histo_bins),
						rwidth=0.75, histtype='bar', normed=True, log=True)
	plt.title(r'histo: orderparam $R(t_{end})$ of each osci over all realizations')
	plt.xlabel(r'R($t_{end}$)')
	plt.ylabel(r'loghist[$R(t_{end})$]')
	plt.savefig('results/hist_orderparam_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

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
	plt.title(r't=0: mean phase $\bar{\phi}=%.3f$ and std $\sigma_{\phi}=%.5f$  [Hz]' %( np.mean(phi[:,int(round(delay/dt))+1,:].flatten()), np.mean(np.std(phi[:,int(round(delay/dt))+1,:], axis=1),axis=0) ) )
	plt.xlabel(r'$\phi(t=0,TSim)$')
	plt.ylabel(r'loghist$[\phi(t=0,TSim)]$')
	plt.savefig('results/histo_phases_t0_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	plt.figure('histo: phases over all oscis at t={2,4,6,8}*delay shifted to same mean')	# plot a histogram of the last frequency of each oscillators over all realizations
	plt.clf()
	# print('mean phase at t = 8*tau', np.mean( phi[:,int(round(8*delay/dt)),:].flatten() ) )
	common_bins=np.linspace( ( np.mean( phi[:,int(round(8*delay/dt)),:].flatten() ) - 4.0*np.std( phi[:,int(round(8*delay/dt)),:].flatten() ) ),
							 ( np.mean( phi[:,int(round(8*delay/dt)),:].flatten() ) + 4.0*np.std( phi[:,int(round(8*delay/dt)),:].flatten() ) ),
							 num=histo_bins)
	shift_mean1 = np.mean( phi[:,int(round((8.*delay)/dt)),:].flatten() ) - np.mean( phi[:,int(round((2.*delay)/dt)),:].flatten() )
	shift_mean2 = np.mean( phi[:,int(round((8.*delay)/dt)),:].flatten() ) - np.mean( phi[:,int(round((4.*delay)/dt)),:].flatten() )
	shift_mean3 = np.mean( phi[:,int(round((8.*delay)/dt)),:].flatten() ) - np.mean( phi[:,int(round((6.*delay)/dt)),:].flatten() )
	plt.hist( ( phi[:,int(round((2.*delay)/dt)),:].flatten() + shift_mean1),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.9, label='t=2*tau')
	plt.hist( ( phi[:,int(round((4.*delay)/dt)),:].flatten() + shift_mean2),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=4*tau')
	plt.hist( ( phi[:,int(round((6.*delay)/dt)),:].flatten() + shift_mean3),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.6, label='t=6*tau')
	plt.hist( phi[:,int(round((8.*delay)/dt)-10),:].flatten(),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.45, label='t=8*tau')
	plt.legend()
	plt.title(r't=$8\tau$: mean phase $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(phi[:,int(round((8.*delay)/dt)-10),:].flatten())/(2.0*np.pi), np.mean(np.std(phi[:,int(round((8.*delay)/dt)-10),:], axis=1), axis=0)/(2.0*np.pi) ) )
	plt.xlabel(r'phase bins [rad]')
	plt.ylabel(r'$\log[hist(\phi)]$')
	plt.savefig('results/histo_phases_all_osci_diff_times_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	plt.figure('histo: last instant. freq. over all oscis at t=TSim')			# plot a histogram of the last frequency of each oscillators over all realizations
	plt.clf()
	plt.hist(lastfreq[:,:,:].flatten(),
						bins=np.linspace( ( np.mean(lastfreq[:,:,:].flatten())-4.0*np.std(lastfreq[:,:,:].flatten()) ), ( np.mean(lastfreq[:,:,:].flatten())+4.0*np.std(lastfreq[:,:,:].flatten()) ),
						num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar')  #, color=(1, np.random.rand(1), np.random.rand(1)) )
	plt.title(r't=TSim: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( np.mean(lastfreq[:,0,:].flatten())/(2.0*np.pi), np.mean(np.std(lastfreq[:,0,:], axis=1),axis=0)/(2.0*np.pi) ) )
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'$loghist[\dot{\phi}(t=TSim)]$')
	plt.savefig('results/histo_lastfreq_all_TSim_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	plt.figure('histo: instant. freq. over all oscis at t={0.1,TSim}')			# plot a histogram of the last frequency of each oscillators over all realizations
	plt.clf()
	common_bins = np.linspace(	min( np.array([firstfreqsext[:,int(round((delay+.1)/dt)),:].flatten(), lastfreq[:,:,:].flatten()]).flatten() ),
							  	max( np.array([firstfreqsext[:,int(round((delay+.1)/dt)),:].flatten(), lastfreq[:,:,:].flatten()]).flatten() ),
								num=histo_bins )
	plt.hist(firstfreqsext[:,int(round((delay+.1)/dt)),:].flatten(), bins=common_bins,
						rwidth=0.75, normed=True, log=True, histtype='bar', label='t=0.1')
	plt.hist(lastfreq[:,:,:].flatten(), bins=common_bins,
						rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=TSim')
	plt.legend()
	plt.title(r't=$0.1$: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( np.mean(firstfreqsext[:,int(round((delay+.1)/dt)),:].flatten()/(2.0*np.pi)), np.mean(np.std(firstfreqsext[:,int(round((delay+.1)/dt)),:], axis=1), axis=0)/(2.0*np.pi) ) )
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'loghist[$\dot{\phi}$]')
	plt.savefig('results/histo_freq_all_osci_t0p1_tTSim_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	plt.figure('histo: instant. freq. over all oscis at t={2,4,6,8}*delay')		# plot a histogram of the last frequency of each oscillators over all realizations
	plt.clf()
	common_bins = np.linspace( 	min( np.array([firstfreqsext[:,int(round((2.*delay)/dt)),:].flatten(), firstfreqsext[:,int(round((8.*delay)/dt)),:].flatten()]).flatten() ),
								max( np.array([firstfreqsext[:,int(round((2.*delay)/dt)),:].flatten(), firstfreqsext[:,int(round((8.*delay)/dt)),:].flatten()]).flatten() ),
								num=histo_bins )
	plt.hist(firstfreqsext[:,int(round((2.*delay)/dt)),:].flatten(), bins=common_bins,
	 					rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.9, label='t=2*tau')
	plt.hist(firstfreqsext[:,int(round((4.*delay)/dt)),:].flatten(), bins=common_bins,
						rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=4*tau')
	plt.hist(firstfreqsext[:,int(round((6.*delay)/dt)),:].flatten(), bins=common_bins,
						rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.6, label='t=6*tau')
	plt.hist(firstfreqsext[:,int(round((8.*delay)/dt)-10),:].flatten(), bins=common_bins,
						rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.45, label='t=8*tau')
	plt.legend()
	plt.title(r't=$8\tau$: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.5f$  [Hz]' %( np.mean(firstfreqsext[:,int(round((8.*delay)/dt)-10),:].flatten())/(2.0*np.pi), np.mean(np.std(firstfreqsext[:,int(round((8.*delay)/dt)-10),:], axis=1), axis=0)/(2.0*np.pi) ) )
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'$loghist[\dot{\phi}]$')
	plt.savefig('results/histo_freq_all_osci_diff_times_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	''' TIME EVOLUTION '''

	plt.figure('order parameter time series for each realization and mean')
	plt.clf()
	for i in range (orderparam.shape[0]):
		# print('Plot order parameter of realization #', i)
		plt.plot(t*dt,orderparam[i,:], alpha=0.2)
	plt.plot(t*dt,np.mean(orderparam[:,:], axis=0), label='mean order param', color='r')
	plt.plot(delay, orderparam[0,int(round(delay/dt))], 'yo', ms=5)# mark where the simulation starts
	plt.title(r'order parameter, with 1-$\bar{R}$=%.3e, and std=%.3e' %(1-np.mean(orderparam[-int(2*1.0/(F*dt)):]), np.std(orderparam[-int(2*1.0/(F*dt)):])), fontdict=titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict=labelfont)
	plt.ylabel(r'$R( t,m = %d )$' % k, fontdict=labelfont)
	plt.legend(loc='center right')
	plt.savefig('results/orderParams-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/orderParams-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure('time evolution of mean and standard deviation of phases, first and all realization')
	plt.subplot(2,1,1)
	plt.title('time evolution of mean and std of phases')
	plt.plot(t*dt, np.mean(phi[0,:,:], axis=1), label=r'$\langle \phi_{first} \rangle$')
	plt.plot(t*dt, np.mean(phi[:,:,:], axis=tuple(range(0, 3, 2))), label=r'$\langle \phi_{all} \rangle$')
	plt.ylabel(r'$\bar{\phi}$', fontdict = labelfont)
	plt.legend(loc='upper left')
	plt.subplot(2,1,2)
	plt.plot(t*dt, np.std(phi[0,:,:], axis=1), label=r'$\sigma_{\phi,first}$')
	plt.plot(t*dt, np.mean(np.std(phi[:,:,:], axis=2), axis=0), label=r'$\sigma_{\phi,all}$')
	plt.legend(loc='upper left')
	plt.xlabel(r'$t$ $[s]$', fontdict=labelfont)
	plt.ylabel(r'$\sigma_{\phi}$', fontdict=labelfont)
	plt.savefig('results/mean_of_phase_vs_t_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	plt.figure('time evolution of mean and standard deviation of frequencies, first and all realizations')
	plt.subplot(2,1,1)
	plt.title('time evolution of mean and std of frequencies')
	plt.plot(t[0:-1]*dt, np.mean(np.diff(phi[0,:,:], axis=0)/dt, axis=1), label=r'$\langle \dot{\phi}_{first} \rangle$')
	plt.plot(t[0:-1]*dt, np.mean(np.diff(phi[:,:,:], axis=1)/dt, axis=tuple(range(0, 3, 2))), label=r'$\langle \dot{\phi}_{all} \rangle$')
	plt.legend(loc='best')
	plt.ylabel(r'$\dot{\bar{\phi}}$')
	plt.subplot(2,1,2)
	plt.plot(t[0:-1]*dt, np.std(np.diff(phi[0,:,:], axis=0)/dt, axis=1), label=r'$\sigma_{\dot{\phi},first}$')
	plt.plot(t[0:-1]*dt, np.std(np.diff(phi[:,:,:], axis=1)/dt, axis=tuple(range(0, 3, 2))), label=r'$\sigma_{\dot{\phi},all}$')
	plt.legend(loc='best')
	plt.xlabel(r'$t$ $[s]$', fontdict=labelfont)
	plt.ylabel(r'$\sigma_{\dot{\phi}}$', fontdict=labelfont)
	plt.savefig('results/mean_of_freq_vs_t_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))

	plt.figure('plot phase time-series for first realization')						# plot the phase time-series of all oscis and all realizations
	plt.clf()
	for i in range (phi.shape[2]):
		#print('Plot now osci #', i, '\n')
		plt.plot((t*dt),phi[0,:,i])
	# print(np.arange(0,len(phi),2*np.pi*F_Omeg*dt))
	#plt.plot(phi[:,1]-2*np.pi*dt*F_Omeg*np.arange(len(phi)))				# plots the difference between the expected phase (according to Omega) vs the actual phase evolution
	plt.plot(delay, phi[0,int(round(delay/dt)),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	plt.title(r'time series phases, $\dot{\phi}_0(t_{start})=$%.4f, $\dot{\phi}_0(t_{end})=$%.4f  [rad/Hz]' %( ((phi[0][11][0]-phi[0][1][0])/(10*dt)), ((phi[0][-4][0]-phi[0][-14][0])/(10*dt)) ), fontdict=titlefont)
	plt.xlabel(r'$t$ $[s]$', fontdict=labelfont)
	plt.ylabel(r'$\phi(t)$', fontdict=labelfont)
	plt.savefig('results/phases-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/phases-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.figure('plot instantaneous-frequencies for first realization')		# plot the frequencies of the oscillators over time
	plt.clf()
	for i in range (phi.shape[2]):											# for each oscillator: plot of frequency vs time
		plt.plot((t[0:-1]*dt), np.transpose( np.diff(phi[0,:,i])/dt ))
	phidot = np.diff(phi[:,:,0], axis=1)/dt
	plt.plot(delay-dt, phidot[0,int(round(delay/dt)-1)], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F*dt))]*dt, t[-1]*dt, color='b', alpha=0.3)
	# print(t[-int(2*1.0/(F*dt))], t[-1])
	plt.title(r'mean frequency [rad Hz] of last $2T$-eigenperiods $\bar{f}=$%.4f' % np.mean(phidot[-int(round(2*1.0/(F*dt))):, 0] ))
	plt.xlabel(r't [s]', fontdict = labelfont)
	plt.ylabel(r'$\dot{\phi}(t)$ $[rad Hz]$', fontdict=labelfont)
	plt.savefig('results/freq-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.pdf' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day))
	plt.savefig('results/freq-vs-time_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_%d_%d_%d.png' %(K, Fc, F_Omeg, delay, now.year, now.month, now.day), dpi=dpi_value)

	plt.draw()
	plt.show()
