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

''' EVALUATION SINGLE REALIZATION '''
def plotTimeSeries(phi, F, dt, orderparam, k, delay, F_Omeg, K):
	now = datetime.datetime.now()

	t = np.arange(phi.shape[0])													# plot the phases of the oscillators over time
	plt.figure(9)
	plt.clf()
	plt.plot(phi)
	# print(np.arange(0,len(phi),2*np.pi*F_Omeg*dt))
	#plt.plot(phi[:,1]-2*np.pi*dt*F_Omeg*np.arange(len(phi)))					# plots the difference between the expected phase (according to Omega) vs the actual phase evolution
	plt.plot(int(round(delay/dt)), phi[int(round(delay/dt)),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F*dt))], t[-1], color='b', alpha=0.3)
	plt.title(r'time series phases, $\dot{\phi}_0(t_{start})=%.4f$, $\dot{\phi}_0(t_{end})=%.4f$  [rad/Hz]' %( ((phi[2][0]-phi[1][0])/(dt)), ((phi[-4][0]-phi[-5][0])/(dt)) ) )
	plt.xlabel(r'$t$ in steps $dt$')
	plt.ylabel(r'$\phi(t)$')
	plt.savefig('results/phases-vs-time_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/phases-vs-time_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=200)
	print(r'frequency of zeroth osci at the beginning and end of the simulation:, $\dot{\phi}_0(t_{start})=%.4f$, $\dot{\phi}_0(t_{end})=%.4f$  [rad/Hz]', ((phi[2][0]-phi[1][0])/(dt)), ((phi[-4][0]-phi[-5][0])/(dt)) )
	print('last values of the phases:\n', phi[-3:,:])

	#plot distribution of instantaneous frequencies for time invterval t-2tau bis t

	# print('the last entries of phi_0 or its derivative:', np.diff(phi[-4:, 0]))
	# print('\nphi check dimension, Doppelklammer', phi[-4:][0])				# CAREFUL HERE!!!, not always the same, the first case M=phi[-4:] is returned and then of this M, M[0] is taken
	# print('phi check dimension, Doppelklammer',   phi[-4:, 0], '\n')			# here
	#print('the last entries of phi_0', np.linspace(F-K, F+K, num=20) )

	plt.figure(10)																# plot a histogram of the frequencies of the oscillators over time
	plt.clf()
	lastfreqs = (np.diff(phi[-int(2*1.0/(F*dt)):, :], axis=0).flatten()/(dt))
	plt.hist(lastfreqs, bins=np.linspace(2*np.pi*(F-K), 2*np.pi*(F+K), num=21), rwidth=0.75 )
	plt.xlim((2*np.pi*(F-K), 2*np.pi*(F+K)))
	plt.title(r'mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [rad/Hz]' %( np.mean(lastfreqs)/(2.0*np.pi), np.std(lastfreqs)/(2.0*np.pi) ) )
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'number')
	plt.savefig('results/freq_histo_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	# plot phase modulo 2pi vs time
	# plt.figure(11)
	# plt.clf()
	# plt.plot(np.fmod(phi, 2.0*np.pi))
	# plt.plot(np.fmod(phi[:,1]-2*np.pi*dt*F_Omeg*np.arange(len(phi)),2*np.pi))
	# plt.title(r'time series phases modulo $2\pi$')
	# plt.axvspan(t[-int(2*1.0/(F*dt))], t[-1], color='b', alpha=0.3)
	# plt.xlabel(r'$t$ in steps $dt$')
	# plt.ylabel(r'$\phi(t)$')
	# plt.savefig('results/phases2pi-vs-time_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	# plt.savefig('results/phases2pi-vs-time.png', dpi=150)

	plt.figure(12)																# plot the frequencies of the oscillators over time
	plt.clf()
	phidot = np.diff(phi, axis=0)/dt
	plt.plot(phidot)
	plt.plot(int(round(delay/dt)), phidot[int(round(delay/dt)),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F*dt))], t[-1], color='b', alpha=0.3)
	# print(t[-int(2*1.0/(F*dt))], t[-1])
	plt.title(r'time series frequencies [rad Hz]')								# , last value in [Hz]: f_1 = %.3d' % (phi[-1][0]-phi[-2][0])/dt )
	plt.xlabel(r'$t$ in steps $dt$')
	plt.ylabel(r'$\dot{\phi}(t)$')
	plt.savefig('results/freq-vs-time_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/freq-vs-time_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=200)

	# fig, ax = plt.subplots() 													# create a new figure with a default 111 subplot
	# phidot = np.diff(phi, axis=0)/dt
	# plt.title(r'time series frequencies [rad Hz]')								# , last value in [Hz]: f_1 = %.3d' % (phi[-1][0]-phi[-2][0])/dt )
	# plt.xlabel(r'$t$ in steps $dt$')
	# plt.ylabel(r'$\dot{\phi}(t)$')
	# ax.plot(phidot)
	# plt.plot(int(round(delay/dt)), phidot[int(round(delay/dt)),0], 'yo', ms=5)
	# plt.axvspan(t[-int(2*1.0/(F*dt))], t[-1], color='b', alpha=0.3)
	# axins = zoomed_inset_axes(ax, 5, loc=4) 									# zoom-factor: 2.5, location: 'best': 0, (only implemented for axes legends)
	# 																			# 'upper right': 1, 'upper left'  : 2, 'lower left'  : 3, 'lower right' : 4, 'right' : 5,
	# 																			# 'center left': 6, 'center right': 7, 'lower center': 8, 'upper center': 9, 'center': 10
	# axins.plot(phidot)
	# x1, x2, y1, y2 = t[-int((25/F)/dt)], t[-1], phidot[-1,0]-5*F, phidot[-1,0]+5*F	# specify the limits of the inset
	# print('inset coordinates: x1=', x1, 'x2=', x2, 'y1=', y1, 'y2=', y2)
	# axins.set_xlim(x1, x2) 														# apply the x-limits
	# axins.set_ylim(y1, y2) 														# apply the y-limits
	# plt.yticks(visiblesimulateNetwork=False)
	# plt.xticks(visible=False)
	# mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
	# print(t[-int(2*1.0/(F*dt))], t[-1])

	plt.figure(14)																# plot the order parameter in dependence of time
	plt.clf()
	plt.plot(t, orderparam)
	plt.plot(int(round(delay/dt)), orderparam[int(round(delay/dt))], 'yo', ms=5)# mark where the simulation starts
	plt.title(r'order parameter, with 1-$\bar{R}$=%.3e, and std=%.3e' %(1-np.mean(orderparam[-int(2*1.0/(F*dt)):]), np.std(orderparam[-int(2*1.0/(F*dt)):])) )
	plt.xlabel(r'$t$ in steps $dt$')
	plt.ylabel(r'$R( t,m = %d )$' % k)
	plt.savefig('results/orderParam-vs-time_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/orderParam-vs-time_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=200)
	#print('\nlast entry order parameter: R-1 = %.3e' % (orderparam[-1]-1) )
	#print('\nlast entries order parameter: R = ', orderparam[-25:])

	plt.draw()

''' EVALUATION BRUTE-FORCE BASIN OF ATTRACTION '''
def doEvalBruteForce(Fc, F_Omeg, K, N, k, delay, twistdelta, results, allPoints, initPhiPrime0, paramDiscretization):
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
	plt.title(r'mean $R(t,m)$, constant dim: $\phi_0^{\prime}=%.2f$' % initPhiPrime0)
	plt.xlabel(r'$\phi_1^{\prime}$')
	plt.ylabel(r'$\phi_2^{\prime}$')
	plt.xlim([1.05*allPoints[:,0].min(), 1.05*allPoints[:,0].max()])
	plt.ylim([1.05*allPoints[:,1].min(), 1.05*allPoints[:,1].max()])
	plt.colorbar()
	if N==3:
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
	plt.savefig('results/rot_red_PhaseSpace_meanR_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/rot_red_PhaseSpace_meanR_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=200)

	plt.figure(2)
	plt.clf()
	plt.scatter(allPoints[:,0], allPoints[:,1], c=results[:,1], alpha=0.5, edgecolor='')
	plt.title(r'last $R(t,m)$, constant dim: $\phi_0^{\prime}=%.2f$' % initPhiPrime0)
	plt.xlabel(r'$\phi_1^{\prime}$')
	plt.ylabel(r'$\phi_2^{\prime}$')
	plt.xlim([1.05*allPoints[:,0].min(), 1.05*allPoints[:,0].max()])
	plt.ylim([1.05*allPoints[:,1].min(), 1.05*allPoints[:,1].max()])
	plt.colorbar()
	if N==3:
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
	plt.savefig('results/rot_red_PhaseSpace_lastR_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/rot_red_PhaseSpace_lastR_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=200)

	plt.figure(3)
	plt.clf()
	tempresults = results[:,0].reshape((paramDiscretization, paramDiscretization))   #np.flipud()
	tempresults = np.transpose(tempresults)
	plt.imshow(tempresults, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower', extent=(allPoints[:,0].min(), allPoints[:,0].max(), allPoints[:,1].min(), allPoints[:,1].max()), vmin=0, vmax=1)
	plt.title(r'mean $R(t,m)$, constant dim: $\phi_0^{\prime}=%.2f$' % initPhiPrime0)
	plt.xlabel(r'$\phi_1^{\prime}$')
	plt.ylabel(r'$\phi_2^{\prime}$')
	plt.colorbar()
	if N==3:
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
		plt.xlim([1.05*allPoints[:,0].min(), 1.05*allPoints[:,0].max()])
		plt.ylim([1.05*allPoints[:,1].min(), 1.05*allPoints[:,1].max()])
	plt.savefig('results/imshow_PhaseSpace_meanR_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/imshow_PhaseSpace_meanR_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=1000)

	plt.figure(4)
	plt.clf()
	tempresults = results[:,1].reshape((paramDiscretization, paramDiscretization))   #np.flipud()
	tempresults = np.transpose(tempresults)
	plt.imshow(tempresults, interpolation='nearest', cmap=cm.coolwarm, aspect='auto', origin='lower', extent=(allPoints[:,0].min(), allPoints[:,0].max(), allPoints[:,1].min(), allPoints[:,1].max()), vmin=0, vmax=1)
	plt.title(r'last $R(t,m)$, constant dim: $\phi_0^{\prime}=%.2f$' % initPhiPrime0)
	plt.title(r'last $R(t,m=)$, constant dim: $\phi_0^{\prime}=%.2f$' % initPhiPrime0)
	plt.xlabel(r'$\phi_1^{\prime}$')
	plt.ylabel(r'$\phi_2^{\prime}$')
	plt.colorbar()
	if N==3:
		plt.plot(alltwistPR[:,1],alltwistPR[:,2], 'yo', ms=8)
		plt.xlim([1.05*allPoints[:,0].min(), 1.05*allPoints[:,0].max()])
		plt.ylim([1.05*allPoints[:,1].min(), 1.05*allPoints[:,1].max()])
	plt.savefig('results/imshow_PhaseSpace_lastR_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/imshow_PhaseSpace_lastR_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=1000)

	plt.draw()
	plt.show()

	return 0.0


''' EVALUATION MANY (noisy, dstributed parameters) REALIZATIONS '''
def doEvalManyNoisy(F, Fc, F_Omeg, K, N, k, delay, twistdelta, results, allPoints, dt, orderparam, r, phi, omega_0, K_0):
	orderparam = np.array(orderparam)
	r          = np.array(r)

	#firstfreq  = (np.diff(phi[:,0:1, :], axis=1)/(dt))						# calculate first frequency of each time series (history of simulation): for each oscillator and realization
	firstfreqs    = (np.diff(phi[:,0:int(round(delay/dt))+4, :], axis=1)/(dt))	# calculate first frequencies of each time series: for each oscillator and realization
	firstfreqsext = (np.diff(phi[:,0:int(round((8*delay)/dt)), :], axis=1)/(dt))
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
	np.savez('results/freqs_%d_%d_%d.npz' %(now.year, now.month, now.day), data=[firstfreqsext, firstfreq, lastfreq])

	'''PLOT TEST'''
	dpi_value  = 300
	histo_bins = 75

	# mpl.rcParams['mathtext.fontset'] = 'custom'
	# mpl.rcParams['mathtext.rm'] = 'Helvetica'
	# mpl.rcParams['mathtext.it'] = 'Helvetica:italic'
	# mpl.rcParams['mathtext.bf'] = 'Helvetica:bold'
	t = np.arange(phi.shape[1])												# plot the phases of the oscillators over time, create "time vector"

	print('plot data:\n')
	''' HISTOGRAMS PHASES AND FREQUENCIES '''

	plt.figure('histo: orderparameter at t=TSim over all realizations') 	# plot the distribution of instantaneous frequencies of the history
	plt.clf()
	for i in range (phi.shape[0]):											# loop over all realizations
		plt.hist(orderparam[:,-1], bins=np.linspace(0, 1, num=histo_bins), rwidth=0.75, histtype='bar', normed=True, log=True)
	plt.title(r'histo: orderparam $R(t_{end})$ of each osci over all realizations')
	plt.xlabel(r'R($t_{end}$)')
	plt.ylabel(r'hist[$R(t_{end})$]')
	plt.savefig('results/hist_orderparam_TSim_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	plt.figure('histo: intrinsic frequencies of oscillators drawn at setup: omega_k(-delay)')
	plt.clf()
	plt.hist(omega_0.flatten(), bins=np.linspace(2*np.pi*(F-3*K), 2*np.pi*(F+3*K), num=histo_bins), rwidth=0.75, histtype='bar', normed=True, log=True)
	plt.title(r'mean intrinsic frequency $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(omega_0.flatten())/(2.0*np.pi), np.std(omega_0.flatten())/(2.0*np.pi) ) )
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'$g[\omega]$')
	plt.savefig('results/omega0_intfreq_histo_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	plt.figure('histo: phases at t={0,TSim} over all realizations and oscis - shifted dist. at TSim to mean of first')	# plot the distribution of instantaneous frequencies of the history
	plt.clf()
	shift_mean = ( np.mean( phi[:,-1,:].flatten() ) - np.mean( phi[:,int(round(delay/dt))+1,:].flatten() ) ) # distance between the mean values of the distribution at different times
	# common_bins=np.linspace(min( np.array( [ (phi[:,int(round(delay/dt)),:].flatten()+shift_mean), (phi[:,-1,:]) ] ).flatten() ),
	# 						max( np.array( [ (phi[:,int(round(delay/dt)),:].flatten()+shift_mean), (phi[:,-1,:]) ] ).flatten() ),
	# 						num=histo_bins)
	# print(phi[:,int(round(delay/dt))+1,:].flatten() + shift_mean)
	plt.hist(phi[:,int(round(delay/dt))+1,:].flatten()+shift_mean,
			#bins=common_bins,
			color='b', rwidth=0.75, histtype='bar', label='t=0', normed=True, log=True)
	plt.hist( ( phi[:,-1,:].flatten() ),
			#bins=common_bins,
			color='r', rwidth=0.75, histtype='bar', label='t=TSim', normed=True, log=True)
	#plt.xlim((1.1*min(firstfreq), 1.1*max(firstfreq)))
	plt.legend(loc='best')
	plt.title(r't=0: mean phase $\bar{\phi}=%.3f$ and std $\sigma_{\phi}=%.3f$  [Hz]' %( np.mean(phi[:,int(round(delay/dt))+1,:].flatten()), np.mean(np.std(phi[:,int(round(delay/dt))+1,:], axis=1),axis=0) ) )
	plt.xlabel(r'$\phi(t=0,TSim)$')
	plt.ylabel(r'hist$[\phi(t=0,TSim)]$')
	plt.savefig('results/histo_phases_t0_TSim_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	plt.figure('histo: phases over all oscis at t={2,4,6,8}*delay')	# plot a histogram of the last frequency of each oscillators over all realizations
	plt.clf()
	common_bins=np.linspace( min(phi[:,int(round(2*delay/dt)),:].flatten()), max(phi[:,int(round(8*delay/dt)),:].flatten()), num=histo_bins)
	plt.hist(phi[:,int(round((2.*delay)/dt)),:].flatten(),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.9, label='t=2*tau')
	plt.hist(phi[:,int(round((4.*delay)/dt)),:].flatten(),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=4*tau')
	plt.hist(phi[:,int(round((6.*delay)/dt)),:].flatten(),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.6, label='t=6*tau')
	plt.hist(phi[:,int(round((8.*delay)/dt)-10),:].flatten(),
			bins=common_bins, rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.45, label='t=8*tau')
	plt.xlim((2*np.pi*(F-3*K), 2*np.pi*(F+3*K)))
	plt.legend()
	plt.title(r't=$8\tau$: mean phase $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(phi[:,int(round((8.*delay)/dt)-10),:].flatten())/(2.0*np.pi), np.mean(np.std(phi[:,int(round((8.*delay)/dt)-10),:], axis=1), axis=0)/(2.0*np.pi) ) )
	plt.xlabel(r'phase bins [rad]')
	plt.ylabel(r'$\phi$')
	plt.savefig('results/histo_phases_all_osci_diff_times_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	# plt.figure('histo: instant. frequencies t=-delay and t=dt of all oscis over all realizations') 	# plot the distribution of instantaneous frequencies of the history
	# plt.clf()
	# plt.hist(firstfreq[:,:].flatten(), bins=np.linspace(2*np.pi*(F-K), 2*np.pi*(F+K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar', label='t = -delay')
	# plt.hist(simstartfreq[:,:].flatten(), bins=np.linspace(2*np.pi*(F-K), 2*np.pi*(F+K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar', label='t = dt')
	# plt.legend(loc='best')
	# #plt.xlim((1.1*min(firstfreq), 1.1*max(firstfreq)))
	# plt.title(r't=dt: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(simstartfreq[:,:].flatten())/(2.0*np.pi), np.mean(np.std(simstartfreq[:,:], axis=1),axis=0)/(2.0*np.pi) ) )
	# plt.xlabel(r'frequency bins [rad/s]')
	# plt.ylabel(r'hist$[\dot{\phi}]$')
	# plt.savefig('results/histo_-taufreq_dt-freq_tau_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	plt.figure('histo: last instant. freq. over all oscis at t=TSim')		# plot a histogram of the last frequency of each oscillators over all realizations
	plt.clf()
	plt.hist(lastfreq[:,:,:].flatten(), bins=np.linspace(2*np.pi*(F-3*K), 2*np.pi*(F+3*K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar')  #, color=(1, np.random.rand(1), np.random.rand(1)) )
	plt.title(r't=TSim: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(lastfreq[:,0,:].flatten())/(2.0*np.pi), np.mean(np.std(lastfreq[:,0,:], axis=1),axis=0)/(2.0*np.pi) ) )
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'$\dot{\phi}(t=TSim)$')
	plt.savefig('results/histo_lastfreq_all_TSim_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	plt.figure('histo: instant. freq. over all oscis at t={0.1,TSim}')		# plot a histogram of the last frequency of each oscillators over all realizations
	plt.clf()
	plt.hist(firstfreqsext[:,int(round((delay+.1)/dt)),:].flatten(), bins=np.linspace(2*np.pi*(F-3*K), 2*np.pi*(F+3*K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar', label='t=0.1')
	plt.hist(lastfreq[:,:,:].flatten(), bins=np.linspace(2*np.pi*(F-3*K), 2*np.pi*(F+3*K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=TSim')
	plt.xlim((2*np.pi*(F-3*K), 2*np.pi*(F+3*K)))
	plt.legend()
	plt.title(r't=$0.1$: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(firstfreqsext[:,int(round((delay+.1)/dt)),:].flatten()/(2.0*np.pi)), np.mean(np.std(firstfreqsext[:,int(round((delay+.1)/dt)),:], axis=1), axis=0)/(2.0*np.pi) ) )
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'hist[$\dot{\phi}$]')
	plt.savefig('results/histo_freq_all_osci_t0p1_tTSim_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	plt.figure('histo: instant. freq. over all oscis at t={2,4,6,8}*delay')	# plot a histogram of the last frequency of each oscillators over all realizations
	plt.clf()
	plt.hist(firstfreqsext[:,int(round((2.*delay)/dt)),:].flatten(), bins=np.linspace(2*np.pi*(F-3*K), 2*np.pi*(F+3*K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.9, label='t=2*tau')
	plt.hist(firstfreqsext[:,int(round((4.*delay)/dt)),:].flatten(), bins=np.linspace(2*np.pi*(F-3*K), 2*np.pi*(F+3*K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.75, label='t=4*tau')
	plt.hist(firstfreqsext[:,int(round((6.*delay)/dt)),:].flatten(), bins=np.linspace(2*np.pi*(F-3*K), 2*np.pi*(F+3*K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.6, label='t=6*tau')
	plt.hist(firstfreqsext[:,int(round((8.*delay)/dt)-10),:].flatten(), bins=np.linspace(2*np.pi*(F-3*K), 2*np.pi*(F+3*K), num=histo_bins), rwidth=0.75, normed=True, log=True, histtype='bar', alpha=0.45, label='t=8*tau')
	plt.xlim((2*np.pi*(F-3*K), 2*np.pi*(F+3*K)))
	plt.legend()
	plt.title(r't=$8\tau$: mean frequency $\bar{f}=%.3f$ and std $\sigma_f=%.3f$  [Hz]' %( np.mean(firstfreqsext[:,int(round((8.*delay)/dt)-10),:].flatten())/(2.0*np.pi), np.mean(np.std(firstfreqsext[:,int(round((8.*delay)/dt)-10),:], axis=1), axis=0)/(2.0*np.pi) ) )
	plt.xlabel(r'frequency bins [rad/s]')
	plt.ylabel(r'$\dot{\phi}$')
	plt.savefig('results/histo_freq_all_osci_diff_times_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	''' TIME EVOLUTION '''

	plt.figure('order parameter time series for each realization and mean')
	plt.clf()
	for i in range (orderparam.shape[0]):
		# print('Plot order parameter of realization #', i)
		plt.plot(t,orderparam[i,:], alpha=0.2)
	plt.plot(t,np.mean(orderparam[:,:], axis=0), label='mean order param', color='r')
	plt.plot(int(round(delay/dt)), orderparam[0,int(round(delay/dt))], 'yo', ms=5)# mark where the simulation starts
	plt.title(r'order parameter, with 1-$\bar{R}$=%.3e, and std=%.3e' %(1-np.mean(orderparam[-int(2*1.0/(F*dt)):]), np.std(orderparam[-int(2*1.0/(F*dt)):])) )
	plt.xlabel(r'$t$ in steps $dt$')
	plt.ylabel(r'$R( t,m = %d )$' % k)
	plt.legend(loc='center right')
	plt.savefig('results/orderParams-vs-time_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/orderParams-vs-time_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=dpi_value)

	plt.figure('time evolution of mean and standard deviation of phases, first and all realization')
	plt.subplot(2,1,1)
	plt.title('time evolution of mean and std of phases')
	plt.plot(t, np.mean(phi[0,:,:], axis=1), label=r'$\langle \phi_{first} \rangle$')
	plt.plot(t, np.mean(phi[:,:,:], axis=tuple(range(0, 3, 2))), label=r'$\langle \phi_{all} \rangle$')
	plt.ylabel(r'$\bar{\phi}$')
	plt.legend(loc='upper left')
	plt.subplot(2,1,2)
	plt.plot(t, np.std(phi[0,:,:], axis=1), label=r'$\sigma_{\phi,first}$')
	plt.plot(t, np.mean(np.std(phi[:,:,:], axis=2), axis=0), label=r'$\sigma_{\phi,all}$')
	plt.legend(loc='upper left')
	plt.xlabel(r'time in time steps dt=%.3f' %dt)
	plt.ylabel(r'$\sigma_{\phi}$')
	plt.savefig('results/mean_of_phase_vs_t_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	plt.figure('time evolution of mean and standard deviation of frequencies, first and all realizations')
	plt.subplot(2,1,1)
	plt.title('time evolution of mean and std of frequencies')
	plt.plot(t[0:-1], np.mean(np.diff(phi[0,:,:], axis=0)/dt, axis=1), label=r'$\langle \dot{\phi}_{first} \rangle$')
	plt.plot(t[0:-1], np.mean(np.diff(phi[:,:,:], axis=1)/dt, axis=tuple(range(0, 3, 2))), label=r'$\langle \dot{\phi}_{all} \rangle$')
	plt.legend(loc='best')
	plt.ylabel(r'$\dot{\bar{\phi}}$')
	plt.subplot(2,1,2)
	plt.plot(t[0:-1], np.std(np.diff(phi[0,:,:], axis=0)/dt, axis=1), label=r'$\sigma_{\dot{\phi},first}$')
	plt.plot(t[0:-1], np.std(np.diff(phi[:,:,:], axis=1)/dt, axis=tuple(range(0, 3, 2))), label=r'$\sigma_{\dot{\phi},all}$')
	plt.legend(loc='best')
	plt.xlabel(r'time in time steps dt=%.3f' %dt)
	plt.ylabel(r'$\sigma_{\dot{\phi}}$')
	plt.savefig('results/mean_of_freq_vs_t_%d_%d_%d.pdf' %(now.year, now.month, now.day))

	plt.figure('plot phase time-series for first realization')						# plot the phase time-series of all oscis and all realizations
	plt.clf()
	for i in range (phi.shape[2]):
		#print('Plot now osci #', i, '\n')
		plt.plot(t,phi[0,:,i])
	# print(np.arange(0,len(phi),2*np.pi*F_Omeg*dt))
	#plt.plot(phi[:,1]-2*np.pi*dt*F_Omeg*np.arange(len(phi)))				# plots the difference between the expected phase (according to Omega) vs the actual phase evolution
	plt.plot(int(round(delay/dt)), phi[0,int(round(delay/dt)),0], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F*dt))], t[-1], color='b', alpha=0.3)
	plt.title(r'time series phases, $\dot{\phi}_0(t_{start})=%.4f$, $\dot{\phi}_0(t_{end})=%.4f$  [rad Hz]' %( ((phi[0][11][0]-phi[0][1][0])/(10*dt)), ((phi[0][-4][0]-phi[0][-14][0])/(10*dt)) ) )
	plt.xlabel(r'$t$ in steps $dt$')
	plt.ylabel(r'$\phi(t)$')
	plt.savefig('results/phases-vs-time_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/phases-vs-time_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=dpi_value)

	plt.figure('plot instantaneous-frequencies for first realization')		# plot the frequencies of the oscillators over time
	plt.clf()
	for i in range (phi.shape[2]):											# for each oscillator: plot of frequency vs time
		plt.plot(np.transpose( np.diff(phi[0,:,i])/dt ))
	phidot = np.diff(phi[:,:,0], axis=1)/dt
	plt.plot(int(round(delay/dt)), phidot[0,int(round(delay/dt))], 'yo', ms=5)
	plt.axvspan(t[-int(2*1.0/(F*dt))], t[-1], color='b', alpha=0.3)
	# print(t[-int(2*1.0/(F*dt))], t[-1])
	plt.title(r'time series frequencies [rad Hz]')							# , last value in [Hz]: f_1 = %.3d' % (phi[-1][0]-phi[-2][0])/dt )
	plt.xlabel(r'$t$ in steps $dt$')
	plt.ylabel(r'$\dot{\phi}(t)$')
	plt.savefig('results/freq-vs-time_%d_%d_%d.pdf' %(now.year, now.month, now.day))
	plt.savefig('results/freq-vs-time_%d_%d_%d.png' %(now.year, now.month, now.day), dpi=dpi_value)

	plt.draw()
	plt.show()
