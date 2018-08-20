#!/usr/bin/python

from __future__ import division
from __future__ import print_function

import sys
import simulation as sim
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, freeze_support
import itertools
from itertools import permutations as permu
from itertools import combinations as combi
import matplotlib
import os
if not os.environ.get('SGE_ROOT') == None:										# this environment variable is set within the queue network, i.e., if it exists, 'Agg' mode to supress output
	print('NOTE: \"matplotlib.use(\'Agg\')\"-mode active, plots are not shown on screen, just saved to results folder!\n')
	matplotlib.use('Agg') #'%pylab inline'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import time
import datetime
import scipy
from scipy import signal

''' All plots in latex mode '''
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def rotate_phases(phi0, isInverse=False):
	''' Rotates the phases such that the phase space direction phi_0 is rotated onto the main diagonal of the n dimensional phase space

		Author: Daniel Platz

	Parameters
	----------
	phi  :  np.array
			array of phases
	isInverse  :  bool
				  if True: returns coordinates of physical phase space in terms of the rotated coordinate system
				  (implies that isInverse=True gives you the coordinates in the rotated system)

	Returns
	-------
	phi_0_rotated  :  np.array
					  phases in rotated or physical phase space '''

	# Determine rotation angle
	n = len(phi0)
	if n <= 1:
		print('ERROR, 1d value cannot be rotated!')

	alpha = -np.arccos(1.0 / np.sqrt(n))

	# Construct rotation matrix
	v = np.zeros((n, n))
	v[0, 0] = 1.0
	v[1:, 1:] = 1.0 / float(n - 1)
	w = np.zeros((n, n))
	w[1:, 0] = -1.0 / np.sqrt(n - 1)
	w[0, 1:] = 1.0 / np.sqrt(n - 1)
	r = np.identity(n) + (np.cos(alpha) - 1) * v + np.sin(alpha) * w			# for N=3 --> 
	# print('---------------------------------------')
	# print('---------------------------------------')
	# print(v)
	# print('---------------------------------------')
	# print(w)
	# print('---------------------------------------')
	# print(r)

	# Apply rotation matrix
	if not isInverse:															# if isInverse==False, this condition is True
		return np.dot(r, phi0)													# transform input into rotated phase space
	else:
		return np.dot(np.transpose(r), phi0)									# transform back into physical phase space

def get_dphi_matrix(n):
	m = np.zeros((n * (n - 1), n))
	x0 = 0
	x1 = 0
	for i in range(n * (n - 1)):
		x0 = int(np.floor(i / float(n - 1)))
		m[i, x0] = 1
		if x1 == x0:
			x1 += 1
			x1 = np.mod(x1, n)

		m[i, x1] = -1
		x1 += 1
		x1 = np.mod(x1, n)
	return m


def get_d_matrix(n):
	'''Constructs a matrix to compute the phase differences from a 
	   vector of non-rotated phases'''
	d = np.zeros((n, n))
	for i in range(n):
		d[i, i] = -1
		d[i, np.mod(i + 1, n)] = 1
	return d

class PhaseDifferenceCell(object):
	def __init__(self, n):
		self.n = n
		self.dphi_matrix = get_dphi_matrix(n)
		self.d_min = -np.pi
		self.d_max = np.pi
		self.d = get_d_matrix(n)

	def is_inside(self, x, isRotated=False):
		# Check if vector has the correct length
		if len(x) != self.n:
			raise Exception('Vector has not the required length n.')

		# Rotate back to initial coordinate system if required
		if isRotated:
			x_tmp = rotate_phases(x, isInverse=False)
		else:
			x_tmp = x

		# Map to phase difference space
		dphi = np.dot(self.d, x_tmp)

		is_inside = True
		for i in range(len(dphi) - 1):
			if np.abs(dphi[i]) > self.d_max:
				is_inside = False
				break

		return is_inside


	def is_inside_old(self, x, isRotated=False):
		'''Checks if a vector is inside the phase difference unit cell.

		Parameters
		----------
		x  :  np.array
 				coordinate vector of length n which is the number of non-reduced dimensions
		isRotated  :  boolean
 						True if x is given in rotated coordinates

		Returns
		-------
		is_inside  :  boolean
 						True if point is inside unit cell
		'''
		# Check if vector has the correct length
		if len(x) != self.n:
			raise Exception('Vector has not the required length n.')

		# Rotate back to initial coordinate system if required
		if isRotated:
			x_tmp = rotate_phases(x, isInverse=False)
		else:
			x_tmp = x

		# Map to phase difference space
		d = np.dot(self.dphi_matrix, x_tmp)

		is_inside = True
		for i in range(len(d)):
			if d[i] < self.d_min:
				is_inside = False
				break
			if d[i] > self.d_max:
				is_inside = False
				break

		return is_inside




''' CALCULATE KURAMOTO ORDER PARAMETER '''
def calcKuramotoOrderParameter(phi):
	'''Computes the Kuramoto order parameter r for in-phase synchronized states

	   Parameters
	   ----------
	   phi:  np.array
	 		real-valued 2d matrix or 1d vector of phases
	 		in the 2d case the columns of the matrix represent the individual oscillators

	   Returns
	   -------
	   r  :  np.array
	 		real value or real-valued 1d vetor of the Kuramotot order parameter

	   Authors
	   -------
	   Lucas Wetzel, Daniel Platz'''
	# Complex phasor representation
	z = np.exp(1j * phi)

	# Kuramoto order parameter
	if len(phi.shape) == 1:
		r = np.abs(np.mean(z))
	elif len(phi.shape) == 2:
		r = np.abs(np.mean(z, axis=1))
	else:
		print( 'Error: phi with wrong dimensions' )
		r = None

	return r


def mTwistOrderParameter(phi):
	'''Computes the Fourier order parameter 'rm' for all m-twist synchronized states

	   Parameters
	   ----------
	   phi  :  np.array
			   real-valued 2d matrix or 1d vector of phases
			   in the 2d case the columns of the matrix represent the individual oscillators

	   Returns
	   -------
	   rm  :  np.array
			  complex-valued 1d vector or 2d matrix of the order parameter

	   Authors
	   -------
	   Lucas Wetzel, Daniel Platz'''

	# Complex phasor representation
	zm = np.exp(1j * phi)

	# Fourier transform along the oscillator index axis
	if len(phi.shape) == 1:
		rm = np.fft.fft(zm) / len(phi)
	elif len(phi.shape) == 2:
		rm = np.fft.fft(zm, axis=1) / phi.shape[1]
	else:
		print('Error: phi with wrong dimensions')
		rm = None
	return rm


def _CheckerboardOrderParameter(phi):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi is supposed
	   to be 1d vector of phases without time evolution.
	'''
	r = 0.0
	for ix in range(len(phi)):
		r += np.exp(1j * phi[ix]) * np.exp(-1j * np.pi * ix)
	r = np.abs(r) / float(len(phi))
	return r


def CheckerboardOrderParameter(phi):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi can be a 1d or 2d vector whose first index
	   corresponds to different times.
	   :rtype: np.ndarray
	'''
	if len(phi.shape) == 1:
		return _CheckerboardOrderParameter(phi)
	else:
		r = np.zeros(phi.shape[0])
		for it in range(phi.shape[0]):
			r[it] = _CheckerboardOrderParameter(phi[it, :])
		return r


def _mTwistOrderParameter2d(phi, nx, ny):
	'''Computes the 2d twist order parameters for 2d states. Phi is supposed
	   to be 1d vector of phases. The result is returned as an array of shape (ny, nx)
	'''
	phi_2d = np.reshape(phi, (ny, nx))
	r = np.fft.fft2(np.exp(1j * phi_2d))
	return np.abs(r) / float(len(phi))


def mTwistOrderParameter2d(phi, nx, ny):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi can be a 1d or 2d vector whose first index
	   corresponds to different times.
	'''
	if len(phi.shape) == 1:
		return _mTwistOrderParameter2d(phi, nx, ny)
	else:
		r = []
		for it in range(phi.shape[0]):
			r.append(_mTwistOrderParameter2d(phi[it, :], nx ,ny))
		return np.array(r)


def _CheckerboardOrderParameter2d(phi, nx, ny):
	'''Computes the 2d checkerboard order parameters for 2d states. Phi is supposed
	   to be 1d vector of phases. Please note that there are three different checkerboard states in 2d.
	'''
	k = np.array([[0, np.pi], [np.pi, 0], [np.pi, np.pi]])
	r = np.zeros(3, dtype=np.complex)
	phi_2d = np.reshape(phi, (ny, nx))
	for ik in range(3):
		for iy in range(ny):
			for ix in range(nx):
				r[ik] += np.exp(1j * phi_2d[iy, ix]) * np.exp(-1j * (k[ik, 0] * iy + k[ik, 1] * ix))
	r = np.abs(r) / float(len(phi))
	return r


def CheckerboardOrderParameter2d(phi, nx, ny):
	'''Computes the 1d checkerboard order parameters for 1d states. Phi can be a 1d or 2d vector whose first index
	   corresponds to different times.
	'''
	if len(phi.shape) == 1:
		return _CheckerboardOrderParameter2d(phi, nx, ny)
	else:
		r = []
		for it in range(phi.shape[0]):
			r.append(CheckerboardOrderParameter2d(phi[it, :], nx, ny))
		return np.array(r)


def oracle_mTwistOrderParameter(phi, k):  # , kx, ky
	'''Computes the absolute value of k-th Fourier order parameter 'rm' for all m-twist synchronized states

	   Parameters
	   ----------
	   phi: np.array
			 real-valued 2d matrix or 1d vector of phases
			 in the 2d case the columns of the matrix represent the individual oscillators
	   k  : integer
			 the index of the requested Fourier order parameter

	   Returns
	   -------
	   rm  : np.complex/np.array
			   real value/real-valued 1d vector of the k-th order parameter

	   Authors
	   -------
	   Lucas Wetzel, Daniel Platz'''

	r = mTwistOrderParameter(phi)
	if len(phi.shape) == 1:
		rk = np.abs(r[k])
	elif len(phi.shape) == 2:
		rk = np.abs(r[:, k])
	else:
		print('Error: phi with wrong dimensions')
		rk = None
	return rk


def oracle_CheckerboardOrderParameter1d(phi, k=1):
	"""
		k == 0 : global sync state
		k == 1 : checkerboard state
	"""
	if k == 0:
		return calcKuramotoOrderParameter(phi)
	elif k == 1:
		return CheckerboardOrderParameter(phi)
	else:
		raise Exception('Non-valid value for k')


def oracle_mTwistOrderParameter2d(phi, nx, ny, kx, ky):
	return mTwistOrderParameter2d(phi, nx, ny)[:, ky, kx]


def oracle_CheckerboardOrderParameter2d(phi, nx, ny, k):
	"""
			k == 0 : x checkerboard state
			k == 1 : y checkerboard state
			k == 2 : xy checkerboard state
			k == 3 : global sync state
		"""
	if k == 0 or k == 1 or k == 2:
		return CheckerboardOrderParameter2d(phi, nx, ny)[:, k]
	elif k == 3:
		return calcKuramotoOrderParameter(phi)
	else:
		raise Exception('Non-valid value for k')


''' CALCULATE SPECTRUM '''
def calcSpectrum(phi,Fsample,waveform=None,decayTimeSlowestMode=None):
	Pxx_db=[]; f=[];
	windowset='boxcar' #'hamming'
	print('current window option is', windowset, 'for waveform', waveform)
	window = scipy.signal.get_window(windowset, int(Fsample), fftbins=True)		# choose window from: boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall,
																				# barthann, kaiser (needs beta), gaussian (needs std), general_gaussian (needs power, width), slepian (needs width), chebwin (needs attenuation)
	print('calculate spectrum for signals with waveform:', waveform, 'and cut the beginning 25percent of the time-series. Implement better solution using decay times.')
	phisteps = len(phi[0,:,0])													# determine length of time-series of phi, then only analyze the part without the transients
	analyzeL = int(0.75 * phisteps)
	for i in range ( len(phi[0,0,:]) ):
		tsdata = generateOscillationSignal(phi[0,-analyzeL:,i],waveform=waveform)
		ftemp, Pxx = scipy.signal.periodogram(tsdata, Fsample, return_onesided=True, window=windowset, axis=0)
		Pxx_db.append( 10*np.log10(Pxx) )
		f.append( ftemp )

	# ma = np.ma.masked_inside(f_db,0,2)										# mask spectrum

	# print('f_db:', f_db, 'Pxx_db:', Pxx_db)
	# return f_db[ma.mask], Pxx_db[ma.mask]
	return f, Pxx_db

''' GENERATE OSCILLATION SIGNAL -- a function to call different coupling functions'''
def generateOscillationSignal(phi,waveform):
	if waveform == 'square':
		return scipy.signal.square(phi,duty=0.5)
	elif waveform == 'sin':
		return np.sqrt(2)*np.sin(phi)
	elif waveform == 'cos':
		return np.sqrt(2)*np.cos(phi)

''' GET FILTER STATUS IN SYNCHRONISED STATE '''
def getFilterStatus(F,K,Fc,delay,Fsim,Tsim):
	dt = 1.0/Fsim
	Nsteps = int(Tsim*Fsim)
	delay_steps = int(delay/dt)
	pll_list = [ PhaseLockedLoop(
					Delayer(delay,dt),
					PhaseDetectorCombiner(idx_pll,[(idx_pll+1)%2]),
					LowPass(Fc,dt,y=0),
					VoltageControlledOscillator(F,K,dt,c=0,phi=0)
					)  for idx_pll in range(2) ]
	_  = simulatePhaseModel(Nsteps,2,pll_list)
	return pll_list[0].lf.y

''' MODEL FITTING: DEMIR MODEL '''
def fitModelDemir(f_model,d_model,fitrange=0):

	f_peak = f_model[np.argmax(d_model)]										# find main peak

	if fitrange != 0:															# mask data
		ma = np.ma.masked_inside(f_model,f_peak-fitrange,f_peak+fitrange)
		f_model_ma = f_model[ma.mask]
		d_model_ma = d_model[ma.mask]
	else:
		f_model_ma = f_model
		d_model_ma = d_model

	A = np.sqrt(2)																# calculate power of main peak for sine wave
	P_offset = 10*np.log10(A**2/2)

	optimize_func = lambda p: P_offset + 10*np.log10( (p[0]**2 * p[1])/(np.pi * p[0]**4 * p[1]**2 + (f_model_ma-p[0])**2 )) # model fit
	error_func = lambda p: optimize_func(p) - d_model_ma
	p_init = (f_peak,1e-8)
	p_final,success = leastsq(error_func,p_init[:])

	f_model_ma = f_model														# restore data
	d_model_ma = d_model

	return f_model, optimize_func(p_final), p_final
