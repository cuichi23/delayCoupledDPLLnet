#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys, gc
import numpy as np
#cimport numpy as np
#cimport cython
import networkx as nx
from scipy.signal import sawtooth
from scipy.stats import cauchy

import matplotlib
import matplotlib.pyplot as plt
import evaluation as eva
import datetime
import time

''' Enable automatic carbage collector '''
gc.enable();

#%%cython --annotate -c=-O3 -c=-march=native

''' CLASSES

authors: Alexandros Pollakis, Lucas Wetzel [ lwetzel@pks.mpg.de ]
'''

# PLL
class PhaseLockedLoop:
	"""A phase-locked loop class"""
	def __init__(self,delayer,pdc,lf,vco):										# sets PLL properties as given when an object of this class is created (see line 253), where pll_list is created
		self.delayer = delayer
		self.pdc = pdc
		self.lf = lf
		self.vco = vco

	def next(self,idx_time,phi):
		x, x_delayed = self.delayer.next(idx_time,phi)							# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb = self.pdc.next(x, x_delayed, idx_time)							# the phase detector signal is computed
		x_ctrl, cLF_t = self.lf.next(x_comb)									# the filtering at the loop filter is applied to the phase detector signal
		phi = self.vco.next(x_ctrl)[0]											# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def nextrelax(self,idx_time,phi):
		x, x_delayed = self.delayer.next(idx_time,phi)							# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb = self.pdc.next(x, x_delayed, idx_time)							# the phase detector signal is computed
		x_ctrl, cLF_t = self.lf.nextrelax(x_comb)								# the filtering at the loop filter is applied to the phase detector signal
		phi = self.vco.next(x_ctrl)[0]											# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def nextadiab(self,idx_time,phi):
		x, x_delayed = self.delayer.next(idx_time,phi)							# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb = self.pdc.next(x, x_delayed, idx_time)							# the phase detector signal is computed
		x_ctrl, cLF_t = self.lf.nextadiab(x_comb)								# the filtering at the loop filter is applied to the phase detector signal
		phi = self.vco.next(x_ctrl)[0]											# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def nextrelaxVCOnoise(self,idx_time,phi):
		x, x_delayed = self.delayer.next(idx_time,phi)							# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb = self.pdc.next(x, x_delayed, idx_time)							# the phase detector signal is computed
		x_ctrl, cLF_t = self.lf.next(x_comb)									# the filtering at the loop filter is applied to the phase detector signal
		phi = self.vco.nextrelax(x_ctrl)[0]										# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def nextadiabVCOnoise(self,idx_time,phi):
		x, x_delayed = self.delayer.next(idx_time,phi)							# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb = self.pdc.next(x, x_delayed, idx_time)							# the phase detector signal is computed
		x_ctrl, cLF_t = self.lf.next(x_comb)									# the filtering at the loop filter is applied to the phase detector signal
		phi = self.vco.nextadiab(x_ctrl)[0]										# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def nextrelaxKt(self,idx_time,phi):
		x, x_delayed = self.delayer.next(idx_time,phi)							# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb = self.pdc.next(x, x_delayed, idx_time)							# the phase detector signal is computed
		x_ctrl, cLF_t = self.lf.next(x_comb)									# the filtering at the loop filter is applied to the phase detector signal
		phi = self.vco.nextrelax(x_ctrl)[0]										# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def nextadiabKt(self,idx_time,phi):
		x, x_delayed = self.delayer.next(idx_time,phi)							# this gets the values of a signal x at time t, and time t-tau, from the delayer (x is the matrix phi here)
		x_comb = self.pdc.next(x, x_delayed, idx_time)							# the phase detector signal is computed
		x_ctrl, cLF_t = self.lf.next(x_comb)									# the filtering at the loop filter is applied to the phase detector signal
		phi = self.vco.nextadiab(x_ctrl)[0]										# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def setup_hist(self):														# set the initial history of the PLLs, all evolving with frequency F_Omeg, provided on program start
		phi = self.vco.set_initial()[0]											# [0] vector-entry zero of the two values saved to phi which is returned
		return phi

	def set_delta_pertubation(self,idx_time,phi,phiS,inst_Freq):
		x_ctrl = self.lf.set_initial_control_signal(phi,inst_Freq)				# the filtering at the loop filter is applied to the phase detector signal
		# print('x =', x, ', x_delayed =', x_delayed, ', x_comb =', x_comb, ', x_ctrl =', x_ctrl)
		phi = self.vco.delta_perturbation(phi,phiS,x_ctrl)[0]
		return phi

	def next_magic(self):														# this is the closed-loop case of the free running PLL
		x_ctrl = self.lf.y														# set control signal; here: lf.y = 0 at the beginning
		phi = self.vco.next(x_ctrl)[0]											# [0] vector-entry zero of the two values returned
		return phi

	def next_free(self):														# evolve phase of PLLs as if they were uncoupled, hence x_ctrl = 0.0, i.e., open-loop PLL --> VCO
		phi = self.vco.next(0.0)[0]
		return phi

# LF: y = integral du x(t) * p(t-u)												# this can be expressed in terms of a differential equation with the help of the Laplace transform
class LowPass:
	"""A lowpass filter class"""
	def __init__(self,Fc,dt,K,F_Omeg,F,cLF=0,Trelax=0,y=0,y_old=0):
		self.Fc = Fc															# set cut-off frequency
		self.wc = 2.0*np.pi*self.Fc												# angular cut-off frequency of the loop filter for a=1, filter of first order
		self.F  = F																# intrinsic frequency of VCO - here needed for x_k^C(0)
		self.F_Omeg = F_Omeg													# provide freq. of synchronized state under investigation - here needed for x_k^C(0)
		self.sOmeg  = 2.0*np.pi*F_Omeg
		self.inst_Freq = 0														# instantaneous frequency
		# here should be the value of K drawn for the VCO from the gaussian distribution, however it might also suffice to use the mean,
		# since the history is given as the perfectly synched state... that does not imply that the K become the same however!
		self.K_Hz	= K
		# self.Kvco_Hz = 2.0 * self.K_Hz
		self.K 	  	= 2.0*np.pi*K												# provide coupling strength - here needed for x_k^C(0)
		# self.Kvco    = 2.0 * K													# Kvco in [rad * Hz]
		self.cLF	 = cLF														# sets the variance of the noise process on the control signal sigma^2=2cLF
		self.cLF_t	 = cLF														# time dependent case (adiabatic change)
		self.d_xcrtl = 0 #-23
		self.dt      = dt														# set time-step
		self.Trelax  = Trelax
		self.Nrelax  = int(self.Trelax/self.dt)
		# print('self.cLF, self.cLF_t', self.cLF, self.cLF_t)

		# if cLF>0:
		# 	print('Simulate system WITH LF noise!')
		# else:
		# 	print('Simulate system WITHOUT LF noise!')
		''' TEST CASES '''
		# self.beta = (dt*2.0*np.pi*Fc) / (dt*2.0*np.pi*Fc + 1)
		# self.beta = (dt*Fc) / (dt*Fc + 1)
		# self.beta = (dt*self.Fc)
		self.beta = self.dt*self.wc

		self.y = y																# denotes the control signal, output of the LF

	def set_initial_control_signal(self,phi,inst_Freq):							# set the control signal for the last time step of the history, in the case the history is a synched state
		self.inst_Freq = inst_Freq												# calculate the instantaneous frequency for the last time step of the history
		#print('\ninstantaneous Frequency to calculate intial filter state: ', self.inst_Freq)
		#self.y = (self.F_Omeg - self.F) / (self.K)								# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
		if self.K!=0:															# this if-call is fine, since it will only be evaluated once
			self.y = (self.inst_Freq - self.F) / (self.K_Hz)					# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
			if self.d_xcrtl == -23:
				self.y = 0.1
			#print('initial control signal x_ctrl=', self.y)

			# print('{self.inst_Freq,self.F,self.K_Hz,controlSignal}:',self.inst_Freq,self.F,self.K_Hz,self.y)

			# self.y = (2.0 * np.pi * (self.inst_Freq - self.F)) / (self.K)
		else:
			self.y = 0.0
		return self.y

	def next(self,xPD):										      				# this updates y=x_k^{C}(t), the control signal, using the input x=x_k^{PD}(t), the phase-detector signal
		y_ctrl0 = 0.0															# since y_ctrl0 = x_k^C(t=0) * \delta(t-0), t=0 is set in function above already!
		# self.y = (1-self.beta)*self.y + self.beta*(xPD-y_ctrl0/self.Fc)			# the difference to the old version is the non-zero initial condition x_k^C(t=0)=[\dot{\theta}_k(0)-\omega] / K
		self.y = (1.0-self.beta)*self.y + self.beta*xPD
		# print('deterministic state of filter AFTER update:', self.y, ' with input relations xPD=', xPD)
		return self.y, 0.0

class NoisyLowPass(LowPass):
	def set_initial_control_signal(self,phi,inst_Freq):							# set the control signal for the last time step of the history, in the case the history is a synched state
		self.inst_Freq = inst_Freq												# calculate the instantaneous frequency for the last time step of the history
		#print('\ninstantaneous Frequency to calculate intial filter state: ', self.inst_Freq)
		#self.y = (self.F_Omeg - self.F) / (self.K)								# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
		if self.K!=0:															# this if-call is fine, since it will only be evaluated once
			self.y = (self.inst_Freq - self.F) / (self.K_Hz)					# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
																				# NOTE: here we use K_Hz, since we have speficied the frequencies in Hz as well!
			self.y = self.y + np.random.normal(loc=0.0, scale=self.Fc*np.sqrt(2*self.cLF*self.dt))
			#print('\n\n CHECK HERE AGAIN THE DIMENSION FOR THE Fc FACTOR TO THE NOISE TERM - RAD/s or 1/s?, also K_Hz! \n')
			# print('initial control signal x_ctrl=', self.y)
			# self.y = (2.0 * np.pi * (self.inst_Freq - self.F)) / (self.K)
		else:
			self.y = 0.0
		return self.y

	def next(self,xPD):							      							# this updates y=x_k^{C}(t), the control signal, using the input x=x_k^{PD}(t), the phase-detector signal
		y_ctrl0 = 0.0															# since y_ctrl0 = x_k^C(t=0) * \delta(t-0), t=0 is set in function above already!
		# self.y = (1-self.beta)*self.y + self.beta*(xPD-y_ctrl0/self.Fc)			# the difference to the old version is the non-zero initial condition x_k^C(t=0)=[\dot{\theta}_k(0)-\omega] / K
		self.y = (1.0-self.beta)*self.y + self.beta*xPD + np.random.normal(loc=0.0, scale=self.Fc*np.sqrt(2*self.cLF*self.dt))
		# print('state of filter AFTER update:', self.y)
		return self.y, self.cLF

class NoisyLowPassAdiabaticC(LowPass):
	def set_initial_control_signal(self,phi,inst_Freq):							# set the control signal for the last time step of the history, in the case the history is a synched state
		self.inst_Freq = inst_Freq												# calculate the instantaneous frequency for the last time step of the history
		print('\ninstantaneous Frequency to calculate intial filter state: ', self.inst_Freq)
		#self.y = (self.F_Omeg - self.F) / (self.K)								# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
		if self.K!=0:															# this if-call is fine, since it will only be evaluated once
			self.y = (self.inst_Freq - self.F) / (self.K_Hz)					# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
																				# NOTE: here we use K_Hz, since we have speficied the frequencies in Hz as well!
			self.y = self.y + np.random.normal(loc=0.0, scale=self.Fc*np.sqrt(2*self.cLF_t*self.dt))
			# print('initial control signal x_ctrl=', self.y)
			# self.y = (2.0 * np.pi * (self.inst_Freq - self.F)) / (self.K)
		else:
			self.y = 0.0
		return self.y

	def nextrelax(self,xPD):						      						# this updates y=x_k^{C}(t), the control signal, using the input x=x_k^{PD}(t), the phase-detector signal
		y_ctrl0 = 0.0															# since y_ctrl0 = x_k^C(t=0) * \delta(t-0), t=0 is set in function above already!
		# self.y = (1-self.beta)*self.y + self.beta*(xPD-y_ctrl0/self.Fc)			# the difference to the old version is the non-zero initial condition x_k^C(t=0)=[\dot{\theta}_k(0)-\omega] / K
		self.y = (1.0-self.beta)*self.y + self.beta*xPD + np.random.normal(loc=0.0, scale=self.Fc*np.sqrt(2*self.cLF_t*self.dt))
		# print('self.cLF_t:', self.cLF_t)
		# print('state of filter AFTER update:', self.y)
		return self.y, self.cLF_t

	def nextadiab(self,xPD):						      						# this updates y=x_k^{C}(t), the control signal, using the input x=x_k^{PD}(t), the phase-detector signal
		y_ctrl0 = 0.0															# since y_ctrl0 = x_k^C(t=0) * \delta(t-0), t=0 is set in function above already!
		# self.y = (1-self.beta)*self.y + self.beta*(xPD-y_ctrl0/self.Fc)			# the difference to the old version is the non-zero initial condition x_k^C(t=0)=[\dot{\theta}_k(0)-\omega] / K
		self.cLF_t = self.cLF_t - 0.1 / float(self.Trelax/self.dt)
		# print('self.cLF_t:', self.cLF_t)
		self.y     = (1.0-self.beta)*self.y + self.beta*xPD + np.random.normal(loc=0.0, scale=self.Fc*np.sqrt(2*self.cLF_t*self.dt))
		# print('state of filter AFTER update:', self.y)
		return self.y, self.cLF_t

# VCO: d_phi / d_t = omega + K * x
class VoltageControlledOscillator:
	"""A voltage controlled oscillator class"""
	def __init__(self,F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c=None,Trelax=0,K_adiab_r=0,phi=None):
		self.sOmeg = 2.0*np.pi*F_Omeg											# set angular frequency of synchronized state under investigation (sOmeg)
		self.diffconstK = diffconstK
		self.domega = domega
		self.Fc = Fc

		gaussian = 0;

		if domega != 0.0:
			if gaussian == 1:
				self.F = np.random.normal(loc=F, scale=np.sqrt(2.0*domega))		# set intrinsic frequency of the VCO plus gaussian dist. random variable from a distribution
				#print('\nIntrinsic frequencies are Gaussian distributed with diffussion coeffcient: ', domega)
			elif gaussian ==0:
				self.F = cauchy.rvs(loc=F, scale=domega, size=1)				# set intrinsic frequency of the VCO plus gaussian dist. random variable from a distribution
				#print('\nIntrinsic frequencies are Lorentzian distributed with scale parameter: ', domega)
			self.omega = 2.0*np.pi*self.F										# set intrinsic angular frequency of the VCO plus gaussian dist. random variable from a distribution
			#print('Intrinsic freq. from gaussian/lorentzian dist.:', self.omega, 'for diffusion constant/scale parameter domega:', self.domega)
		else:
			self.omega = 2.0*np.pi*F											# set intrinsic frequency of the VCO
		if diffconstK != 0:														# set input sensitivity of VCO [ok to do here, since this is only called when the PLL objects are created]
			self.K = np.random.normal(loc=K, scale=np.sqrt(2.0*diffconstK))		# provide coupling strength - here needed for x_k^C(0)
			self.K = 2.0*np.pi*self.K											# now in [rad * Hz]
			# self.Kvco = 2.0 * self.K
			# print('2*pi*K from gaussian dist.:', self.K, 'for diffusion constant diffconstK:', self.diffconstK)
		else:
			self.K = 2.0*np.pi*K
			# self.Kvco = 2.0 * self.K
		if histtype == 'syncstate':												# set initial frequency according to the parameter in 1params.txt
			self.init_freq = self.sOmeg
		elif histtype == 'uncoupled':
			self.init_freq = self.omega
		else:
			print('\nPROBLEM! set intial condition in 1params.txt correctly!')
		self.Kt 		= self.K												# introduce self.Kt, a coupling strength variable to be adiabatically changed
		self.K_adiab_r 	= 2*np.pi*K_adiab_r										# coupling strength at which the adiabatic change reverses
		self.K_range 	= self.K_adiab_r - self.K
		self.Kt_counter = 0
		self.dt 		= dt													# set time step with which the equations are evolved
		self.phi 		= phi													# this is the internal representation of phi, NOT the container in simulateNetwork
		self.c 			= c														# noise strength -- chose something like variance or std here!
		self.ct			= c
		self.Trelax		= float(Trelax)
		if self.Trelax > 0:
			self.K_rate		= 0.1 / self.Trelax									# rate of change of K per second during adiabatic change
			self.Treverse	= float(self.K_range/self.K_rate)					# time in seconds until the adiabatic change of K reverses
		else:
			self.K_rate 	= 0
			self.Treverse 	= 0
		# print('\nin constructor VCO, c=', c)
		''' CAN OUTPUT RANDOM NUMBERS TO FILE FOR CHECK '''
		# if c==8.0:
		# 	now = datetime.datetime.now()
		# 	rand_numb_test = np.random.normal(loc=0.0, scale=np.sqrt(2.0*c), size=10000)
		# 	np.savez('results/randnumb_test_c%.7e_%d_%d_%d.npz' %(c, now.year, now.month, now.day), rand_numb_test=rand_numb_test)

	def next(self,x_ctrl):														# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
		self.d_phi = self.omega + self.K * x_ctrl
		#print('self.dphi=', self.d_phi)
		#time.sleep(1)
		self.phi = self.phi + self.d_phi * self.dt
		# print('phi:', phi)
		return self.phi, self.d_phi

	def delta_perturbation(self, phi, phiS, x_ctrl):							# sets a delta-like perturbation 0-dt, the last time-step of the history
		self.d_phi = phiS + ( self.omega + self.K * x_ctrl ) * self.dt
		self.phi = self.phi + self.d_phi
		return self.phi, self.d_phi

	def set_initial(self):														# sets the phase history of the VCO with the frequency of the synchronized state under investigation
		# print('sOmeg:', self.sOmeg)
		self.d_phi = self.init_freq * self.dt
		self.phi = self.phi + self.d_phi
		return self.phi, self.d_phi

# + noise
class NoisyVoltageControlledOscillator(VoltageControlledOscillator):			# make a child class of VoltageControlledOscillator (inherits all functions of VoltageControlledOscillator)
	"""A voltage controlled oscillator class WITH noise (GWN)"""
	def next(self,x_ctrl):														# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
																				# watch the separation of terms for order dt and the noise with order sqrt(dt)
		# self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + ( 2.0 * np.pi ) * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.F)) * np.sqrt(self.dt) # scales with self.F
		self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.dt)) # no scaling of noise with frequency
		self.phi = self.phi + self.d_phi
		# print('Difference noise term placement: ', tempphi-self.phi, '\n')
		return self.phi, self.d_phi

	def delta_perturbation(self, phi, phiS, x_ctrl):							# sets a delta-like perturbation 0-dt, the last time-step of the history
		self.d_phi = phiS + ( self.omega + self.K * x_ctrl ) * self.dt
		#+ 2. * np.pi * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*)) * np.sqrt(*self.dt) this can be added only if the diffusion constant is normalized such that
		# for changing tau (length of history) the diffusion of phases is the same - i.e. scale by sqrt(tau)
		self.phi = self.phi + self.d_phi
		return self.phi, self.d_phi

	def set_initial(self):														# sets the phase history of the VCO with the frequency of the synchronized state under investigation
		self.d_phi = self.init_freq * self.dt
		#+ 2.*pi*np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c)) * np.sqrt(*self.dt) this can be added only if the diffusion constant is normalized such that
		# for changing tau (length of history) the diffusion of phases is the same - i.e. scale by sqrt(tau)
		self.phi = self.phi + self.d_phi
		#print('write history with noise')
		return self.phi, self.d_phi

class NoisyVoltageControlledOscillatorAdiabaticChangeC(VoltageControlledOscillator):
	"""A voltage controlled oscillator class WITH noise (GWN) that can be changed adiabatically"""
	def nextrelax(self,x_ctrl):													# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
																				# watch the separation of terms for order dt and the noise with order sqrt(dt)
		# self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + ( 2.0 * np.pi ) * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.F)) * np.sqrt(self.dt) # scales with self.F
		# print('self.Trelax, self.ct',self.Trelax, self.ct)
		self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.ct*self.dt)) # no scaling of noise with frequency
		self.phi = self.phi + self.d_phi
		# print('Difference noise term placement: ', tempphi-self.phi, '\n')
		return self.phi, self.d_phi

	def nextadiab(self,x_ctrl):													# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
																				# watch the separation of terms for order dt and the noise with order sqrt(dt)
		# self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + ( 2.0 * np.pi ) * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.F)) * np.sqrt(self.dt) # scales with self.F
		self.ct = self.ct - 0.1 / float(self.Trelax/self.dt)
		# print('self.Trelax, self.ct, self.dt, self.omega, self.K, self.phi, x_ctrl',self.Trelax, self.ct, self.dt, self.omega, self.K, self.phi, x_ctrl)
		self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.ct*self.dt)) # no scaling of noise with frequency
		self.phi = self.phi + self.d_phi
		# print('Difference noise term placement: ', tempphi-self.phi, '\n')
		return self.phi, self.d_phi

	def delta_perturbation(self, phi, phiS, x_ctrl):							# sets a delta-like perturbation 0-dt, the last time-step of the history
		# print('self.Trelax, self.ct',self.Trelax, self.ct)
		self.d_phi = phiS + ( self.omega + self.K * x_ctrl ) * self.dt
		#+ 2. * np.pi * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*)) * np.sqrt(*self.dt) this can be added only if the diffusion constant is normalized such that
		# for changing tau (length of history) the diffusion of phases is the same - i.e. scale by sqrt(tau)
		self.phi = self.phi + self.d_phi
		return self.phi, self.d_phi

	def set_initial(self):														# sets the phase history of the VCO with the frequency of the synchronized state under investigation
		self.d_phi = self.sOmeg * self.dt
		#+ 2.*pi*np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c)) * np.sqrt(*self.dt) this can be added only if the diffusion constant is normalized such that
		# for changing tau (length of history) the diffusion of phases is the same - i.e. scale by sqrt(tau)
		self.phi = self.phi + self.d_phi
		#print('write history with noise')
		return self.phi, self.d_phi

class NoisyVoltageControlledOscillatorAdiabaticChangeK(VoltageControlledOscillator):
	"""A voltage controlled oscillator class WITH noise (GWN) that can be changed adiabatically"""
	def nextrelax(self,x_ctrl):													# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
																				# watch the separation of terms for order dt and the noise with order sqrt(dt)
		# self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + ( 2.0 * np.pi ) * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.F)) * np.sqrt(self.dt) # scales with self.F
		# print('self.Trelax, self.ct',self.Trelax, self.ct)
		self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.dt)) # no scaling of noise with frequency
		self.phi = self.phi + self.d_phi
		# print('Difference noise term placement: ', tempphi-self.phi, '\n')
		return self.phi, self.d_phi

	def nextadiab(self,x_ctrl):													# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
																				# watch the separation of terms for order dt and the noise with order sqrt(dt)
		# self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + ( 2.0 * np.pi ) * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.F)) * np.sqrt(self.dt) # scales with self.F
		''' first increase K adiabatically, then decrease again '''
		self.Kt = self.Kt + self.K_rate * self.dt * np.sign( int(self.Treverse/self.dt) - self.Kt_counter )
		# if int(self.Treverse/self.dt) < self.Kt_counter:
		# 	print('Adiabatic change of K reversed!')
		# else:
		# 	print('Treverse in steps:', int(self.Treverse/self.dt),'   self.K_range:', self.K_range,'\n')
		# 	time.sleep(5)
		# print('self.Kt=', self.Kt,'\n')
		# print('self.Treverse=', self.Treverse,'\n')
		# print('self.Kt_counter=', self.Kt_counter,'\n')
		# time.sleep(5)

		# print('self.Trelax, self.ct, self.dt, self.omega, self.K, self.phi, x_ctrl',self.Trelax, self.ct, self.dt, self.omega, self.K, self.phi, x_ctrl)
		self.d_phi = ( self.omega + self.Kt * x_ctrl ) * self.dt + np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.dt)) # no scaling of noise with frequency
		self.phi = self.phi + self.d_phi
		# print('Difference noise term placement: ', tempphi-self.phi, '\n')
		self.Kt_counter = self.Kt_counter + 1;
		# print('counter: ',self.Kt_counter,'\n')

		return self.phi, self.d_phi

	def delta_perturbation(self, phi, phiS, x_ctrl):							# sets a delta-like perturbation 0-dt, the last time-step of the history
		# print('self.Trelax, self.ct',self.Trelax, self.ct)
		self.d_phi = phiS + ( self.omega + self.K * x_ctrl ) * self.dt
		#+ 2. * np.pi * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*)) * np.sqrt(*self.dt) this can be added only if the diffusion constant is normalized such that
		# for changing tau (length of history) the diffusion of phases is the same - i.e. scale by sqrt(tau)
		self.phi = self.phi + self.d_phi
		return self.phi, self.d_phi

	def set_initial(self):														# sets the phase history of the VCO with the frequency of the synchronized state under investigation
		self.d_phi = self.sOmeg * self.dt
		#+ 2.*pi*np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c)) * np.sqrt(*self.dt) this can be added only if the diffusion constant is normalized such that
		# for changing tau (length of history) the diffusion of phases is the same - i.e. scale by sqrt(tau)
		self.phi = self.phi + self.d_phi
		#print('write history with noise')
		return self.phi, self.d_phi

# y = 1 / n * sum h( x_delayed_neighbours - x_self )
# print('Phasedetector and Combiner: sawtooth')
class PhaseDetectorCombiner:													# this class creates PD objects, these are responsible to detect the phase differences and combine the results
	"""A phase detector and combiner class"""									# of different inputs (coupling partners)
	def __init__(self,idx_self,idx_neighbours,div=1):
		# print('Phasedetector and Combiner: sawtooth')
		self.h = lambda x: sawtooth(x,width=0.5)								# set the type of coupling function, here a sawtooth since we consider digital PLLs (rectangular signals)
		self.idx_self = idx_self												# assigns the index
		self.div = div															# set division
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11
		# print('Osci ',idx_self,', my neighbors are:', idx_neighbours)
		#print('Initialize PD, neighbors of PLL ',self.idx_self,' are: ', self.idx_neighbours)

	def next(self,x,x_delayed,idx_time=0):										# gets time-series results at delayed time and current time to calculate phase differences
		try:
			# print('x: ', x,'\nx_delayed: ', x_delayed)
			x_self = x[self.idx_self]											# extract own state (phase) at time t and save to x_self
			# if idx_time < 5:
			# 	print('x_self:', x_self)
			# print('neighbors: ', [n for n in self.idx_neighbours], ' type: ', type(self.idx_neighbours))
			if len(self.idx_neighbours) > 0:
				#x_neighbours = x_delayed[self.idx_neighbours]					# extract the states of all coupling neighbors at t-tau and save to x_neighbours
				x_neighbours = x_delayed[self.idx_neighbours]					# extract the states of all coupling neighbors at t-tau and save to x_neighbours
				# print('PLL',self.idx_self,' receives from its neighbors: ', x_neighbours, '  self.idx_neighbours: ', self.idx_neighbours)
				# if idx_time < 5:
				# 	print('x_neighbours:', x_neighbours)
				self.y = np.mean( self.h( ( x_neighbours - x_self )/ self.div ) )		# calculate phase detector output signals and combine them to yield the signal that is fed into the loop filter
				# if idx_time < 5:
				# 	print('x_neighbours.shape:', x_neighbours.shape )
				# 	print('phase detector signal:', self.y)
			else:
				self.y = 0.0;
				#print('Processing free running PLL with control signal:', self.y)
			return self.y
		except:
			print('\n\nCHECK: in PhaseDetectorCombiner set self.idx.neighbors to the iterator for networkx libs > v1.11!\n\n')
			print('NOTE: check again how to set this for XOR, multiplication and flip flop') # if there is no input (uncoupled PLL), then the phase detector output is zero
			print('Also, this has to be set individually for each type of PD!')

			x_self = x[self.idx_self]
			x_neighbours = x_delayed[self.idx_neighbours]
			if len(self.idx_neighbours) > 0:
				self.y = np.mean( self.h( ( x_neighbours - x_self )/ self.div ) )
			else:
				self.y = 0.0;
				# print('Processing free running PLL with control signal:', self.y)

			#sys.exit(1);
			return self.y

class PhaseDetectorCombinerHighFreq(PhaseDetectorCombiner):						# this class creates PD objects, these are responsible to detect the phase differences and combine the results
	"""A phase detector and combiner class"""									# of different inputs (coupling partners)
	def __init__(self,idx_self,idx_neighbours,div=1):
		print('This includes the coupling function of the sum of the phases, note however that this does not reflect the HF-terms for DPLLs!')
		self.h = lambda x: sawtooth(x,width=0.5)								# set the type of coupling function, here a sawtooth since we consider digital PLLs (rectangular signals)
		self.idx_self = idx_self												# assigns the index
		self.div = div															# set division
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11

	def next(self,x,x_delayed,idx_time=0):										# gets time-series results at delayed time and current time to calculate phase differences
		try:
			# print('x: ', x,'\nx_delayed: ', x_delayed)
			x_self = x[self.idx_self]											# extract own state (phase) at time t and save to x_self
			# if idx_time < 5:
			# 	print('x_self:', x_self)
			# print('neighbors: ', [n for n in self.idx_neighbours], ' type: ', type(self.idx_neighbours))
			if len(self.idx_neighbours) > 0:
				#x_neighbours = x_delayed[self.idx_neighbours]					# extract the states of all coupling neighbors at t-tau and save to x_neighbours
				x_neighbours = x_delayed[self.idx_neighbours]					# extract the states of all coupling neighbors at t-tau and save to x_neighbours
				#print('PLL',self.idx_self,' receives from its neighbors: ', x_neighbours, '  self.idx_neighbours: ', self.idx_neighbours)
				# if idx_time < 5:
				# 	print('x_neighbours:', x_neighbours)
				self.y = np.mean( self.h( ( x_neighbours - x_self )/ self.div ) + self.h( ( x_neighbours + x_self )/ self.div ) )	# calculate phase detector output signals and combine them to yield the signal that is fed into the loop filter
				# if idx_time < 5:
				# 	print('x_neighbours.shape:', x_neighbours.shape )
				# 	print('phase detector signal:', self.y)
			else:
				self.y = 0.0;
				#print('Processing free running PLL with control signal:', self.y)
			return self.y
		except:
			print('\n\nCHECK: in PhaseDetectorCombiner set self.idx.neighbors to the iterator for networkx libs > v1.11!\n\n')
			print('NOTE: check again how to set this for XOR, multiplication and flip flop') # if there is no input (uncoupled PLL), then the phase detector output is zero
			print('Also, this has to be set individually for each type of PD!')

			x_self = x[self.idx_self]
			x_neighbours = x_delayed[self.idx_neighbours]
			if len(self.idx_neighbours) > 0:
				self.y = np.mean( self.h( ( x_neighbours - x_self )/ self.div ) + self.h( ( x_neighbours - x_self )/ self.div ) )
			else:
				self.y = 0.0;
				# print('Processing free running PLL with control signal:', self.y)

			#sys.exit(1);
			return self.y

class PhaseDetectorCombinerShifted(PhaseDetectorCombiner):						# this class creates PD objects, these are responsible to detect the phase differences and combine the results
	"""A phase detector and combiner class"""									# of different inputs (coupling partners)
	def __init__(self,idx_self,idx_neighbours):
		# print('Phasedetector and Combiner: sawtooth')
		self.part 	  = 0.95													# this needs to come from the constructor! add/change... 1params.txt content?!
		self.highHarm = 2.0
		self.h = lambda x: sawtooth(x,width=0.5) + self.part * sawtooth(self.highHarm*x + 0.5*np.pi,width=0.5)	# set the type of coupling function, here a sawtooth since we consider digital PLLs (rectangular signals)
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11
		# print('Osci ',idx_self,', my neighbors are:', idx_neighbours)

# y = 1 / n * sum h( x_delayed_neighbours - x_self )
class SinPhaseDetectComb(PhaseDetectorCombiner):								# child class for different coupling function - here sinusoidal
	def __init__(self,idx_self,idx_neighbours):
		# print('Phasedetector and Combiner: sin(x)')
		self.h = lambda x: np.sin(x)											# set the type of coupling function, here a sine-function
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11
		#print('Osci ',idx_self,', my neighbors are:', idx_neighbours)

class SinCosPhaseDetectComb(PhaseDetectorCombiner):								# child class for different coupling function - here sinusoidal and cosinusoidal
	def __init__(self,idx_self,idx_neighbours):
		# print('Phasedetector and Combiner: sin(x)')
		self.part 	  = 0.95													# this needs to come from the constructor! add/change... 1params.txt content?!
		self.highHarm = 2.0
		self.h = lambda x: np.sin(x) + self.part * np.cos(self.highHarm*x)		# set the type of coupling function, here a sine-function
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11
		# print('Osci ',idx_self,', my neighbors are:', idx_neighbours)

# y = 1 / n * sum h( x_delayed_neighbours - x_self )
class CosPhaseDetectComb(PhaseDetectorCombiner):								# child class for different coupling function - here sinusoidal
	def __init__(self,idx_self,idx_neighbours):
		# print('Phasedetector and Combiner: cos(x)')
		self.h = lambda x: np.cos(x)											# set the type of coupling function, here a sine-function
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11

class NoisyPhaseDetectorCombiner(PhaseDetectorCombiner):						# this class creates PD objects, these are responsible to detect the phase differences and combine the results
	"""A phase detector and combiner class with PD phase noise"""				# of different inputs (coupling partners)
	def __init__(self,idx_self,idx_neighbours,dt,cPD=0,div=1):
		# print('Phasedetector and Combiner: sawtooth')
		self.cPD	= cPD														# sets the variance of the noise process on the control signal sigma^2=2cLF
		self.div 	= div														# set division
		self.cPD_t  = cPD														# time dependent case (adiabatic change)
		self.h 		= lambda x: sawtooth(x,width=0.5)							# set the type of coupling function, here a sawtooth since we consider digital PLLs (rectangular signals)
		self.dt		= dt
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11
		print('Initialize PD, neighbors of PLL ',self.idx_self ,' are: ', self.idx_neighbours)

	def next(self,x,x_delayed,idx_time=0):										# gets time-series results at delayed time and current time to calculate phase differences
		try:
			x_self = x[self.idx_self]											# extract own state (phase) at time t and save to x_self
			#x_neighbours = x_delayed[self.idx_neighbours]						# extract the states of all coupling neighbors at t-tau and save to x_neighbours
			# IN TERMS OF ITERATOR
			x_neighbours = x_delayed[self.idx_neighbours]
			if len(self.idx_neighbours) > 0:
				self.y = np.mean( self.h( (( x_neighbours - x_self )/ self.div) + np.random.normal(loc=0.0, scale=np.sqrt(2*self.cPD*self.dt)) ) )	# calculate phase detector output signals and combine them to yield the signal that is fed into the loop filter
			else:
				self.y = 0.0 + np.mean( self.h( np.pi/2.0 + np.random.normal(loc=0.0, scale=np.sqrt(2*self.cPD*self.dt)) ) ); print('This is cheated! Heal before use.')
				# print('Processing free running PLL with control signal:', self.y)
			return self.y
		except:
			print('\n\nCHECK: problem in NoisyPhaseDetectorCombiner!\n\n')
			x_self = x[self.idx_self]											# extract own state (phase) at time t and save to x_self
			x_neighbours = x_delayed[self.idx_neighbours]
			if len(self.idx_neighbours) > 0:
				self.y = np.mean( self.h( (( x_neighbours - x_self )/ self.div) + np.random.normal(loc=0.0, scale=np.sqrt(2*self.cPD*self.dt)) ) )	# calculate phase detector output signals and combine them to yield the signal that is fed into the loop filter
			else:
				self.y = 0.0 + np.mean( self.h( np.pi/2.0 + np.random.normal(loc=0.0, scale=np.sqrt(2*self.cPD*self.dt)) ) ); print('This is cheated! Heal before use.')
				# print('Processing free running PLL with control signal:', self.y)
			self.y = np.mean( self.h( (( x_neighbours - x_self )/ self.div) + np.random.normal(loc=0.0, scale=np.sqrt(2*self.cPD*self.dt)) ) )
			#sys.exit(1);
			return self.y

class NoisyPhaseDetectorCombinerShifted(NoisyPhaseDetectorCombiner):			# this class creates PD objects, these are responsible to detect the phase differences and combine the results
	"""A phase detector and combiner class"""									# of different inputs (coupling partners)
	def __init__(self,idx_self,idx_neighbours,cPD=0):
		# print('Phasedetector and Combiner: sawtooth')
		self.cPD	  = cPD														# sets the variance of the noise process on the control signal sigma^2=2cLF
		self.cPD_t	  = cPD														# time dependent case (adiabatic change)
		self.part 	  = 0.95													# this needs to come from the constructor! add/change... 1params.txt content?!
		self.highHarm = 2.0
		self.h = lambda x: sawtooth(x,width=0.5) + self.part * sawtooth(self.highHarm*x + 0.5*np.pi,width=0.5)	# set the type of coupling function, here a sawtooth since we consider digital PLLs (rectangular signals)
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11
		# print('Osci ',idx_self,', my neighbors are:', idx_neighbours)

# y = 1 / n * sum h( x_delayed_neighbours - x_self )
class NoisySinPhaseDetectComb(NoisyPhaseDetectorCombiner):						# child class for different coupling function - here sinusoidal
	def __init__(self,idx_self,idx_neighbours,cPD=0):
		# print('Phasedetector and Combiner: sin(x)')
		self.cPD	  = cPD														# sets the variance of the noise process on the control signal sigma^2=2cLF
		self.cPD_t	  = cPD														# time dependent case (adiabatic change)
		self.h = lambda x: np.sin(x)											# set the type of coupling function, here a sine-function
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11
		#print('Osci ',idx_self,', my neighbors are:', idx_neighbours)


class NoisySinCosPhaseDetectComb(NoisyPhaseDetectorCombiner):					# child class for different coupling function - here sinusoidal and cosinusoidal
	def __init__(self,idx_self,idx_neighbours,cPD=0):
		# print('Phasedetector and Combiner: sin(x)')
		self.cPD	  = cPD														# sets the variance of the noise process on the control signal sigma^2=2cLF
		self.cPD_t	  = cPD														# time dependent case (adiabatic change)
		self.part 	  = 0.95													# this needs to come from the constructor! add/change... 1params.txt content?!
		self.highHarm = 2.0
		self.h = lambda x: np.sin(x) + self.part * np.cos(self.highHarm*x)		# set the type of coupling function, here a sine-function
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11
		# print('Osci ',idx_self,', my neighbors are:', idx_neighbours)

# y = 1 / n * sum h( x_delayed_neighbours - x_self )
class NoisyCosPhaseDetectComb(NoisyPhaseDetectorCombiner):						# child class for different coupling function - here sinusoidal
	def __init__(self,idx_self,idx_neighbours,cPD=0):
		# print('Phasedetector and Combiner: cos(x)')
		self.cPD	  = cPD														# sets the variance of the noise process on the control signal sigma^2=2cLF
		self.cPD_t	  = cPD														# time dependent case (adiabatic change)
		self.h = lambda x: np.cos(x)											# set the type of coupling function, here a sine-function
		self.idx_self = idx_self												# assigns the index
		if nx.__version__ == '1.11':
			self.idx_neighbours = idx_neighbours								# assigns the neighbors according to the coupling topology
		else:
			self.idx_neighbours = [n for n in idx_neighbours] 					# for networkx > v1.11

# delayer
class Delayer:
	"""A delayer class"""
	def __init__(self,delay,dt,feedback_delay,std_dist_delay):
		# print('Delayer set to identical transmission delays')
		self.std_dist_delay = std_dist_delay
		self.feedback_delay = feedback_delay
		if std_dist_delay != 0:
			self.delay = np.random.normal(loc=delay, scale=std_dist_delay)		# process variation, the delays in the network are gaussian distributed about the mean delay
			print('Transmission delays from gaussian dist.:', self.delay, 'for diffusion constant std_dist_delay:', self.std_dist_delay)
			print('NOTE: at the moment these heterogeneous transmission delays are always such that an oscillator receives equally delayed signals from all its neighbors!')
		else:
			self.delay = delay
			#print('delay set to:', delay)
																				# NOTE: static distribution of transmission delays - ensure that max delay determines the length of the history vector
		self.delay_steps = int(round(self.delay/dt))							# when initialized, the delay in time-steps is set to delay_steps
		if ( self.delay_steps < 1 and self.delay_steps != 0 ):
			print('NOTE: the transmission delay is smaller than the time-step "dt", hence "delay_steps" < 1 and the simulations assumes NO DELAY, adjust FSim!')

		if feedback_delay == 0:
			self.feedback_delay_steps = 0
		else:
			self.feedback_delay_steps = int(round(self.feedback_delay/dt))
			print('NOTE: there is a nonzero feedback-delay specified! tau_f=',self.feedback_delay)
		#print('\ndelay steps:', self.delay_steps, '\n')
		# print('\ndelay steps:', self.delay_steps, '\n')

	def next(self,idx_time,x):
		idx_delayed 		 = idx_time - self.delay_steps						# delayed index for incoming signal is calculated
		idx_feedback_delayed = idx_time - self.feedback_delay_steps				# delayed index for feedback signal is calculated
		#if(idx_time >= (self.delay_steps) and idx_time < (self.delay_steps+2) ):
		#	print('\nidx_delayed', idx_delayed, 'with phi[t]=', x[idx_time,:],'with phi[t-tau]=', x[idx_delayed,:],'at idx_time', idx_time, '\n')
		# print('phases at time t:', np.asarray(x[idx_time,:]), 'phases at time t-tau:', np.asarray(x[idx_delayed,:]))
		# print('idx_time:', idx_time, 'idx_delayed', idx_delayed, 'idx_feedback_delayed', idx_feedback_delayed)
		return np.asarray(x[idx_feedback_delayed,:]), np.asarray(x[idx_delayed,:])			# x is is the time-series from which the values at t-dt and t-tau are returned

# class DistDelayDelayer(Delayer):
# 	"""A delayer class"""
# 	def __init__(self,delay,dt,std_dist_delay):
#
# 		# print('Delayer: FOR THIS CASE YOU HAVE TO FIND A SOLUTION FOR THE FREQUENCIES AND HISTORIES! -- take the mean delay to approximate the global frequency, also return distribution of delays')
# 		# the mean delay is chosen, as well as the width... Gaussian distributed -- for history use Omega associated to mean delay value...
#
# 		if std_dist_delay != 0:
# 			self.delay = np.random.normal(loc=delay, scale=std_dist_delay)		# process variation, the delays in the network are gaussian distributed about the mean delay
# 		else:
# 			self.delay = delay
# 																				# NOTE: static distribution of transmission delays - ensure that max delay determines the length of the history vector
# 		self.delay_steps = int(round(self.delay/dt))							# when initialized, the delay in time-steps is set to delay_steps
# 		#print('\ndelay steps:', self.delay_steps, '\n')
#
# 	def next(self,idx_time,x):
# 		idx_delayed = idx_time - self.delay_steps								# delayed index is calculated
# 		#if(idx_time >= (self.delay_steps) and idx_time < (self.delay_steps+2) ):
# 		#	print('\nidx_delayed', idx_delayed, 'with phi[t]=', x[idx_time,:],'with phi[t-tau]=', x[idx_delayed,:],'at idx_time', idx_time, '\n')
# 		#print('phases at time t:', np.asarray(x[idx_time,:]), 'phases at time t-tau:', np.asarray(x[idx_delayed,:]))
# 		return np.asarray(x[idx_time,:]), np.asarray(x[idx_delayed,:])			# x is is the time-series from which the values at t-dt and t-tau are returned

class DistDelayDelayerWithDynNoise(Delayer):
	"""A delayer class -- dynamically fluctuating transmission delays"""

	def next(self,idx_time,x,std_dist_delay,std_dyn_delay_noise):
		idx_delayed = idx_time - self.delay_steps - int(round(np.random.normal(0.0, scale=std_dyn_delay_noise)/dt)) # delayed index for incoming signal is calculated
		idx_feedback_delayed = idx_time - self.feedback_delay_steps				# delayed index for feedback signal is calculated
		#if(idx_time >= (self.delay_steps) and idx_time < (self.delay_steps+2) ):
		#	print('\nidx_delayed', idx_delayed, 'with phi[t]=', x[idx_time,:],'with phi[t-tau]=', x[idx_delayed,:],'at idx_time', idx_time, '\n')
		#print('phases at time t:', np.asarray(x[idx_time,:]), 'phases at time t-tau:', np.asarray(x[idx_delayed,:]))
		return np.asarray(x[idx_feedback_delayed,:]), np.asarray(x[idx_delayed,:])	# x is is the time-series from which the values at t-dt and t-tau are returned

################################################################################

''' SIMULATE NETWORK '''
def simulateNetwork(mode,div,Nplls,F,F_Omeg,K,Fc,delay,feedback_delay,dt,c,Nsteps,topology,couplingfct,histtype,phiS,phiM,domega,diffconstK,diffconstSendDelay,cPD,Nx=0,Ny=0,Trelax=0,Kadiab_value_r=0):

	# NOTE print('WORK HERE, change from Nsteps+delay_steps to delay_steps container with sequential output to file')

	y0 = 0																		# inital filter status:
	''' for the last step of the initial history, the filter status has to be set if one uses the second order (inertia) type description of the model;
		this avoids performing the integration of the filter and reduces the dependence on an entiry history to a ODE of first order for the control signal;
		it is important however, to get the details associated to this transformation, concerning initial condition for the condtrol signal in the case of the ODE first order
		or instead the continuous history in case of the integration '''
	np.random.seed()															# restart pseudo random-number generator
	if not Nx*Ny == Nplls:
		print('Number of PLLs N needs to be equal to Nx*Ny! Here (Nplls, Nx, Ny)=', Nplls, Nx, Ny, 'Correcting now with Nplls=Nx*Ny.')
		Nplls=Nx*Ny
	# NOTE Generate PLL objects here
	pll_list = generatePllObjects(mode,div,topology,couplingfct,histtype,Nplls,dt,c,delay,feedback_delay,F,F_Omeg,K,Fc,y0,phiM,domega,diffconstK,diffconstSendDelay,Nx,Ny,cPD,Trelax,Kadiab_value_r)	# create object lists of PLL objects of the network

	if diffconstSendDelay != 0:
		all_delay_steps=[]
		for i in range(Nplls):
			all_delay_steps = pll_list[i].delayer.delay_steps
		delay_steps = np.max(all_delay_steps)
		now = datetime.datetime.now()
		np.savez('results/dist_delays_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_diffK%.7e_diffCDelay%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, diffconstK, diffconstSendDelay, now.year, now.month, now.day), delays=all_delay_steps)
		print('distributed transmission delays set, see 1params.txt!\n')
	else:
		delay_steps = pll_list[0].delayer.delay_steps      						# get the number of steps representing the delay at a given time-step from delayer of PLL_0
	#WHY Nsteps + delay_steps? generate a container with 'delay_steps+1' and cycle through it
	phi = np.empty([Nsteps+delay_steps,Nplls])									# prepare container for phase time series
	# here the initial phases of all PLLs in pll_list are copied into the first  entry of the container phi for the phases of the PLLs
	phi[0,:] = [pll.vco.phi for pll in pll_list]
	if (histtype == 'uncoupled' and ( mode == 2 or mode == 1 or mode==0 ) ):

		# Here we need to compensate for drift between the PLLs due to unequal intrinsic frequencies in such a way, that over the time of the history, i.e., the delay,
		# the phase difference due to the deviations from the mean frequency are compensated for

		if ( Nplls==3 and ( mode==2 or mode==1 or mode==0 ) and couplingfct == 'triang' ):
			''' heterogeneous intrinsic frequencies - values, also change below ~line 1194 '''
			#F_intrin=[1.012, 1.070, 1.012]; mean_F_intrin=np.mean(F_intrin);
			mean_F_intrin=np.mean(pll_list[:].vco.omega/(2.0*np.pi));
			# print('\nphiS before correction:', phiS, '   for intrinsic frequencies:', F_intrin);
			for i in range(Nplls):
				F_intrin   = pll_list[i].vco.omega/(2.0*np.pi);
				correction = 2.0*np.pi*(F_intrin-mean_F_intrin)*(delay+3*dt);# the deviation of the frequency from the mean intrinsic frequency determines the extra phase shift until t=0, the +3*dt is for correction, see how hist. is set up
																				# when coupling is turned on
				phiS[i] = phiS[i] - correction;
			# print('\nphiS after correction:', phiS);
		elif ( Nplls==6 and ( mode==2 or mode==1 or mode==0 ) and couplingfct == 'triang' ):
			#F_intrin=[1.012, 1.070, 1.012]; mean_F_intrin=np.mean(F_intrin);
			mean_F_intrin=np.mean(pll_list[:].vco.omega/(2.0*np.pi));
			# print('\nphiS before correction:', phiS, '   for intrinsic frequencies:', F_intrin);
			for i in range(Nplls):
				F_intrin   = pll_list[i].vco.omega/(2.0*np.pi);
				correction = 2.0*np.pi*(F_intrin-mean_F_intrin)*(delay+3*dt);# the deviation of the frequency from the mean intrinsic frequency determines the extra phase shift until t=0, the +3*dt is for correction, see how hist. is set up
																				# when coupling is turned on
				phiS[i] = phiS[i] - correction;
			# print('\nphiS after correction:', phiS);

		phi[0,:] = [pll.set_delta_pertubation(0, phi, phiS[i], pll_list[i].vco.init_freq/(2.0*np.pi)) for i,pll in enumerate(pll_list)]
		# if Nplls>2:
		# 	print('Free running PLL history, initial states with phase difference phi2-phi1 and phi3-phi2:', phi[0,1]-phi[0,0],'    ', phi[0,2]-phi[0,1],'\n')
	else:
		phi[0,:] = [pll.vco.phi for pll in pll_list]

	# print('phi[0,:] ->', phi[0,:])
	omega_0  = [pll.vco.omega for pll in pll_list]								# obtain the randomly distributed (gaussian) values for the intrinsic frequencies
	K_0      = [pll.vco.K for pll in pll_list]									# obtain the randomly distributed (gaussian) values for the coupling strength of the VCO
	delays_0 = [pll.delayer.delay for pll in pll_list]							# obtain the randomly distributed (gaussian) values for the transmission delays
	if cPD > 0:
		cPD_0= [pll.pdc.cPD for pll in pll_list]
	Nrelax	 = [(pll.lf.Nrelax) for pll in pll_list]
	cPD_t    = []																# needed for the cases in which cPD_t is not written!
	Kadiab_t = []
	# print('omega_0 ->', omega_0)
	# this is the for-loop that iterates the system, first the initial conditions is set, then the dynamics are computed
	if delay_steps == 0 and Trelax == 0:
		''' This is for delay_steps == 0, cPD-rate == 0 ''' 					#- this could also be achieved by setting delay_steps = 2 to fit into this evolution scheme! however think and check '''
		print('No delay case, iteration scheme carried out differently.')

		# print('phi[0,:]:',phi[0,:], 'F_Omeg:',F_Omeg,'phi:',phi,'phiS:',phiS)
		phi[0,:] = [pll.set_delta_pertubation(0, phi, phiS[i], F_Omeg) for i,pll in enumerate(pll_list)]  # NOTE: phi[idx_time,:] is already set, save intial condition to container phi

		''' YOU SHOULD CHANGE HERE AND SO ON TO SAVE DATA DURING SIMULATION TO KEEP THE RAM AVAILABLE '''

		# started here, not DONE!!! NOTE
		# file_object = open('phase_time_series_K%.2f_Fc%.2f_FOm%.2f_tau%.2f_c%.7e_cPD%.7e_%d_%d_%d.npz' %(K, Fc, F_Omeg, delay, c, cPD, now.year, now.month, now.day)', 'a')							# open file to write phase time series


		for idx_time in range(Nsteps+delay_steps-1):							# iterate over Nsteps from 0 to "Nsteps" no history, just initial conditions
			phi[idx_time+1,:] = [pll.next(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above

	elif delay_steps > 0  and Trelax == 0:
		''' SET INITIAL HISTORY AND PERTURBATION '''
		for idx_time in range(delay_steps):										# iterate over Nsteps from 0 to "Nsteps + delay_steps" -> of that "delay_steps+1" is history
			''' Important note: the container for the phase variables [phi] is indexed from 0 onwards, i.e., idx_time==delay_steps-1 corresponds to the entry in the container
				with index delay_steps-1, and hence the container then has "delay_steps" number of entries.
				Also note however, that below we always set idx_time+1, i.e., when idx_time==delay_steps-1, idx_time+1==delay_steps-1+1==delay_steps is set and the history is
				complete (delay_steps * dt written in real time).'''
			if idx_time <= (delay_steps-2):										# fill phi entries 1 to "delay_steps-2", note: we set idx_time+1 in the last call at idx_time==delay_steps-2
				#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
				phi[idx_time+1,:] = [pll.setup_hist() for pll in pll_list]		# here the initial phase history is set
				#print('Fequency of uncoupled oscis, (phi[idx_time+1,:]-phi[idx_time,:])/(dt*2*np.pi): ', (phi[idx_time+1,:]-phi[idx_time,:])/(dt*2*np.pi),'\n')
				#print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
			if (idx_time == (delay_steps-1)):										# fill the last entry of the history in phi at delay_steps-1
				# print( '\n\nhere is also STILL A PROBLEM HERE: if there is no perturbation, the history should grow constantly until delay_steps (included)\n')
				#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
				inst_Freq = (phi[idx_time,:]-phi[idx_time-1,:])/(dt*2.0*np.pi)	# calculate all instantaneous frequencies of all PLLs
				#print('instantaneous frequency when delta_perturbation is set (measured from simulated phase): ', inst_Freq)
				# print('self.F_Omeg when perturbation is set: ', F_Omeg, '\n')
				#print('CHECK WHETHER CALCULATED AT THE TIME STEP; HERE FOR CALCULTED: instantaneous frequency TOWARDS last step of history:', inst_Freq)
				#print('number of oscis:', Nplls)
				if histtype == 'syncstate':
					phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, phiS[i], inst_Freq[i]) for i,pll in enumerate(pll_list)]
				elif histtype == 'uncoupled':
					# here we set phiS[i] == 0 for all i, since we set the initial phase value already above!
					#print('\nvalue of the phase difference of between the PLLs at end of history:', phi[idx_time,:])
					phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, 0, inst_Freq[i]) for i,pll in enumerate(pll_list)]
					#print('value of the phase difference of between the PLLs at end of history:', phi[idx_time+1,:],'\n')
					#print('History of uncoupled oscis, phi[0,:] and phi[tau-dt,:],phi[tau,:]: ', phi[0,:],'   ', phi[idx_time,:],'    ', phi[idx_time+1,:],'\n')
					#print('Fequency of uncoupled oscis, (phi[idx_time+1,:]-phi[idx_time,:])/(dt*2*np.pi): ', (phi[idx_time+1,:]-phi[idx_time,:])/(dt*2*np.pi),'\n')
					#print(' uncoupled hist new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
		''' Output to monitor the intitial phases with and without perturbation in original and rotated coordinates '''
		# if Nplls==3:
		# 	print('initial phases original system:', phi[0,:],'\ninitial phases rotated system:', eva.rotate_phases(phi[0,:].flatten(), isInverse=True))
		# 	diff1=phi[0,2]-phi[0,1]; diff2=phi[0,1]-phi[0,0]; diff3=phi[0,0]-phi[0,2];
		# 	print('initial phases differences original system, WITHOUT perturbation (3-2, 2-1, 1-3):', diff1%(2*np.pi), diff2%(2*np.pi), diff3%(2*np.pi),
		# 		'\ninitial phase differences rotated system, WITHOUT perturbation:', eva.rotate_phases(np.array([diff1%(2*np.pi), diff2%(2*np.pi), diff3%(2*np.pi)]), isInverse=True), '\n')
		# 	diff1p=phi[idx_time+1,2]-phi[idx_time+1,1]; diff2p=phi[idx_time+1,1]-phi[idx_time+1,0]; diff3p=phi[idx_time+1,0]-phi[idx_time+1,2];
		# 	print('initial phases differences original system, WITH perturbation (3-2, 2-1, 1-3):', diff1p%(2*np.pi), diff2p%(2*np.pi), diff3p%(2*np.pi),
		# 		'\ninitial phase differences rotated system, WITH perturbation:', eva.rotate_phases(np.array([diff1p%(2*np.pi), diff2p%(2*np.pi), diff3p%(2*np.pi)]), isInverse=True), '\n')
		''' NOW SIMULATE THE SYSTEM AFTER HISTORY IS SET '''
		for idx_time in range(delay_steps,Nsteps+delay_steps-1):
			# if idx_time == delay_steps:
			# 	print('\n\nSIMULATION STARTS HERE, phase histories are set, step:', idx_time, '  the frequencies was: ', (phi[idx_time-1][:]-phi[idx_time-2][:])/dt,'\n')
			# if idx_time < delay_steps+10:
			# 	print('prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:], '   -> inst. frequency: ', (phi[idx_time][:]-phi[idx_time-1][:])/dt )
			phi[idx_time+1,:] = [pll.next(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
			# if idx_time < delay_steps+10:
			# 	print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])

	elif delay_steps == 0 and Trelax > 0:
		if c!=0 and cPD!=0 or Kadiab_value_r!=0 and c!=0 or cPD!=0 and Kadiab_value_r!=0:
			print('Check wether this mode works and makes sense.')
		elif c==0 and cPD==0 and Kadiab_value_r==0:
			print('Cannot change adiabatically from zero to zero, either K, c or cPD should be nonzero!')
		elif c>0 and cPD==0 and Kadiab_value_r==0:
			''' This is for delay_steps == 0, cPD-rate > 0  '''
			print('No delay case, iteration scheme carried out differently.')
			Nrelax 		  = int(Trelax/dt)
			print('Relaxation time in [sec]:', Trelax)
			rate 		  = 0.1 / Trelax
			print('Rate of adiabatic change in [dim_of_cPD/sec]:', rate)
			rate_per_step = 0.1 / Nrelax
			''' use Trelax to calculate the rate and Nsteps until cPD is close to zero '''
			Nsteps  	  = int(0.95 * c / rate_per_step)
			print('Simulation time after relaxation time in [s]:', Nsteps*dt, ' and total time in [s]:', Trelax+Nsteps*dt)
			phi = np.empty([Nsteps+Nrelax+delay_steps,Nplls])					# prepare container for phase time series
			cPD_t = np.empty([Nsteps+Nrelax+delay_steps])						# prepare container for c_LF
			''' SET INITIAL HISTORY AND PERTURBATION '''
			if histtype == 'syncstate':
				phi[0,:] = [pll.set_delta_pertubation(0, phi, phiS[i], F_Omeg) for i,pll in enumerate(pll_list)]  # NOTE: phi[idx_time,:] is already set
			elif histtype == 'uncoupled':
				phi[0,:] = [pll.set_delta_pertubation(0, phi, phiS[i], pll_list[i].vco.init_freq) for i,pll in enumerate(pll_list)]  # NOTE: phi[idx_time,:] is already set


			''' adiabatic case, relaxation time '''
			for idx_time in range(Nrelax):										# iterate over Nsteps from 0 to "Nsteps" no history, just initial conditions
				cPD_t[idx_time]	 = [pll.vco.ct][0]
				# print('relaxation:', [pll.pdc.cPD_t][0])
				phi[idx_time+1,:] = [pll.nextrelaxVCOnoise(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
			''' adiabatic case, adiabatic change '''
			for idx_time in range(Nrelax,Nrelax+Nsteps-1):						# iterate over Nsteps from 0 to "Nsteps" no history, just initial conditions
				cPD_t[idx_time]	 = [pll.vco.ct][0]
				# print('adiabatic:', [pll.pdc.cPD_t][0])
				phi[idx_time+1,:] = [pll.nextadiabVCOnoise(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above

		elif c==0 and Kadiab_value_r==0 and cPD >0:
			''' This is for delay_steps == 0, cPD-rate > 0  '''
			print('No delay case, iteration scheme carried out differently.')
			Nrelax 		  = int(Trelax/dt)
			print('Relaxation time in [sec]:', Trelax)
			rate 		  = 0.1 / Trelax
			print('Rate of adiabatic change in [dim_of_cPD/sec]:', rate)
			rate_per_step = 0.1 / Nrelax
			''' use Trelax to calculate the rate and Nsteps until cPD is close to zero '''
			Nsteps  	  = int(0.95 * cPD / rate_per_step)
			print('Simulation time after relaxation time in [s]:', Nsteps*dt, ' and total time in [s]:', Trelax+Nsteps*dt)
			phi = np.empty([Nsteps+Nrelax+delay_steps,Nplls])					# prepare container for phase time series
			cPD_t = np.empty([Nsteps+Nrelax+delay_steps])						# prepare container for c_LF
			''' SET INITIAL HISTORY AND PERTURBATION '''
			if histtype == 'syncstate':
				phi[0,:] = [pll.set_delta_pertubation(0, phi, phiS[i], F_Omeg) for i,pll in enumerate(pll_list)]  # NOTE: phi[idx_time,:] is already set
			elif histtype == 'uncoupled':
				phi[0,:] = [pll.set_delta_pertubation(0, phi, phiS[i], pll_list[i].vco.init_freq) for i,pll in enumerate(pll_list)]  # NOTE: phi[idx_time,:] is already set

			''' adiabatic case, relaxation time '''
			for idx_time in range(Nrelax):										# iterate over Nsteps from 0 to "Nsteps" no history, just initial conditions
				cPD_t[idx_time]	 = [pll.pdc.cPD_t][0]
				# print('relaxation:', [pll.pdc.cPD_t][0])
				phi[idx_time+1,:] = [pll.nextrelax(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
			''' adiabatic case, adiabatic change '''
			for idx_time in range(Nrelax,Nrelax+Nsteps-1):						# iterate over Nsteps from 0 to "Nsteps" no history, just initial conditions
				cPD_t[idx_time]	 = [pll.pdc.cPD_t][0]
				# print('adiabatic:', [pll.pdc.cPD_t][0])
				phi[idx_time+1,:] = [pll.nextadiab(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above

		elif c==0 and Kadiab_value_r>0 and cPD ==0:
			''' This is for delay_steps == 0, Kchange-rate > 0  '''
			print('No delay case, iteration scheme carried out differently.')
			Nrelax 		  = int(Trelax/dt)
			print('Relaxation time in [sec]:', Trelax)
			rate 		  = 0.1 / Trelax
			print('Rate of adiabatic change in [dim_of_Kadiab_value_r/sec]:', rate)
			rate_per_step = 0.1 / Nrelax
			''' use Trelax to calculate the rate and Nsteps until K_adiab is close to zero --> here need twice the time to increase and decrease '''
			Nsteps  	  = 2 * int( (2.0*np.pi*(Kadiab_value_r - K)) / rate_per_step )
			print('Simulation time after relaxation time in [s]:', Nsteps*dt, ' and total time in [s]:', Trelax+Nsteps*dt)
			phi = np.empty([Nsteps+Nrelax+delay_steps,Nplls])					# prepare container for phase time series
			Kadiab_t = np.empty([Nsteps+Nrelax+delay_steps])					# prepare container for K_adiab
			''' SET INITIAL HISTORY AND PERTURBATION '''
			if histtype == 'syncstate':
				phi[0,:] = [pll.set_delta_pertubation(0, phi, phiS[i], F_Omeg) for i,pll in enumerate(pll_list)]  # NOTE: phi[idx_time,:] is already set
			elif histtype == 'uncoupled':
				phi[0,:] = [pll.set_delta_pertubation(0, phi, phiS[i], pll_list[i].vco.init_freq) for i,pll in enumerate(pll_list)]  # NOTE: phi[idx_time,:] is already set

			''' adiabatic case, relaxation time '''
			for idx_time in range(Nrelax):										# iterate over Nsteps from 0 to "Nsteps" no history, just initial conditions
				Kadiab_t[idx_time]	 = [pll.vco.Kt][0]
				# print('relaxation:', [pll.lf.Kadiab_t][0])
				phi[idx_time+1,:] = [pll.nextrelaxKt(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
			''' adiabatic case, adiabatic change '''
			for idx_time in range(Nrelax,Nrelax+Nsteps-1):						# iterate over Nsteps from 0 to "Nsteps" no history, just initial conditions
				Kadiab_t[idx_time]	 = [pll.vco.Kt][0]
				# print('adiabatic:', [pll.lf.Kadiab_t][0])
				phi[idx_time+1,:] = [pll.nextadiabKt(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
			Kadiab_t[idx_time+1]=Kadiab_t[idx_time]

	elif delay_steps > 0  and Trelax > 0:
		if c!=0 and cPD!=0 or Kadiab_value_r!=0 and c!=0 or cPD!=0 and Kadiab_value_r!=0:
			print('Check wether this mode works and makes sense.')
		elif c==0 and cPD==0 and Kadiab_value_r==0:
			print('Cannot change adiabatically from zero to zero, either c or cPD should be nonzero!')

		elif c>0 and cPD==0 and Kadiab_value_r==0:
			''' This is for delay_steps > 0, cPD-rate == 0, c > 0 '''
			Nrelax 		  = int(Trelax/dt)
			print('Relaxation time in [sec]:', Trelax)
			rate 		  = 0.1 / Trelax
			print('Rate of adiabatic change in [dim_of_cPD/sec]:', rate)
			rate_per_step = 0.1 / Nrelax
			''' use Trelax to calculate the rate and Nsteps until cPD is close to zero '''
			Nsteps  	  = int(0.95 * c / rate_per_step)
			print('Simulation time after relaxation time in [s]:', Nsteps*dt, ' and total time in [s]:', Trelax+Nsteps*dt)
			phi = np.empty([Nsteps+Nrelax+delay_steps,Nplls])					# prepare container for phase time series
			cPD_t = np.empty([Nsteps+Nrelax+delay_steps])						# prepare container for c_LF
			''' SET INITIAL HISTORY AND PERTURBATION '''
			for idx_time in range(delay_steps):									# iterate over Nsteps from 0 to "Nsteps + delay_steps" -> of that "delay_steps+1" is history
				''' Important note: the container for the phase variables [phi] is indexed from 0 onwards, i.e., idx_time==delay_steps-1 corresponds to the entry in the container
					with index delay_steps-1, and hence the container then has "delay_steps" number of entries.
					Also note however, that below we always set idx_time+1, i.e., when idx_time==delay_steps-1, idx_time+1==delay_steps-1+1==delay_steps is set and the history is
					complete (delay_steps * dt written in real time).'''
				if idx_time <= (delay_steps-2):						# fill phi entries 1 to "delay_steps-2", note: we set idx_time+1 in the last call at idx_time==delay_steps-2
					#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
					cPD_t[idx_time]	 = [pll.vco.ct][0]
					phi[idx_time+1,:] = [pll.setup_hist() for pll in pll_list]	# here the initial phase history is set
					#print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
				if idx_time == (delay_steps-1):									# fill the last entry of the history in phi at delay_steps-1
					# print( '\n\nhere is also STILL A PROBLEM HERE: if there is no perturbation, the history should grow constantly until delay_steps (included)\n')
					#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
					inst_Freq = (phi[idx_time,:]-phi[idx_time-1,:])/(dt*2*np.pi)# calculate all instantaneous frequencies of all PLLs
					#print('instantaneous frequency when delta_perturbation is set: ', inst_Freq)
					#print('self.F_Omeg when perturbation is set: ', F_Omeg, '\n')
					#print('CHECK WHETHER CALCULATED AT THE TIME STEP; HERE FOR CALCULTED: instantaneous frequency TOWARDS last step of history:', inst_Freq)
					#print('number of oscis:', Nplls)
					cPD_t[idx_time]	 = [pll.vco.ct][0]
					if histtype == 'syncstate':
						phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, phiS[i], inst_Freq[i]) for i,pll in enumerate(pll_list)]
					elif histtype == 'uncoupled':
						phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, 0, inst_Freq[i]) for i,pll in enumerate(pll_list)]
					#print('new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
			''' NOW SIMULATE THE SYSTEM AFTER HISTORY IS SET '''
			for idx_time in range(delay_steps,Nrelax+delay_steps):
				''' adiabatic case, relaxation time '''
				#if idx_time == delay_steps:
				#	print('\n\nSIMULATION STARTS HERE, phase histories are set\n')
				#if idx_time < delay_steps+10:
				#	print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
				cPD_t[idx_time]	 = [pll.vco.ct][0]
				phi[idx_time+1,:] = [pll.nextrelaxVCOnoise(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
				#if idx_time < delay_steps+10:
				#	print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
			for idx_time in range(Nrelax+delay_steps,Nsteps+Nrelax+delay_steps-1):
				''' adiabatic case, adiabatic change '''
				cPD_t[idx_time]	 = [pll.vco.ct][0]
				phi[idx_time+1,:] = [pll.nextadiabVCOnoise(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above

		elif c==0 and Kadiab_value_r==0 and cPD>0:
			''' This is for delay_steps > 0, cPD-rate > 0  '''
			Nrelax 		  = int(Trelax/dt)
			print('Relaxation time in [sec]:', Trelax)
			rate 		  = 0.1 / Trelax
			print('Rate of adiabatic change in [dim_of_cPD/sec]:', rate)
			rate_per_step = 0.1 / Nrelax
			''' use Trelax to calculate the rate and Nsteps until cPD is close to zero '''
			Nsteps  	  = int(0.95 * cPD / rate_per_step)
			print('Simulation time after relaxation time in [s]:', Nsteps*dt, ' and total time in [s]:', Trelax+Nsteps*dt)
			phi = np.empty([Nsteps+Nrelax+delay_steps,Nplls])					# prepare container for phase time series
			cPD_t = np.zeros([Nsteps+Nrelax+delay_steps])						# prepare container for c_LF
			''' SET INITIAL HISTORY AND PERTURBATION '''
			for idx_time in range(delay_steps):									# iterate over Nsteps from 0 to "Nsteps + delay_steps" -> of that "delay_steps+1" is history
				''' Important note: the container for the phase variables [phi] is indexed from 0 onwards, i.e., idx_time==delay_steps-1 corresponds to the entry in the container
					with index delay_steps-1, and hence the container then has "delay_steps" number of entries.
					Also note however, that below we always set idx_time+1, i.e., when idx_time==delay_steps-1, idx_time+1==delay_steps-1+1==delay_steps is set and the history is
					complete (delay_steps * dt written in real time).'''
				if idx_time <= (delay_steps-2):						# fill phi entries 1 to "delay_steps-2", note: we set idx_time+1 in the last call at idx_time==delay_steps-2
					#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
					cPD_t[idx_time]	 = [pll.pdc.cPD_t][0]
					phi[idx_time+1,:] = [pll.setup_hist() for pll in pll_list]		# here the initial phase history is set
					#print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
				if idx_time == (delay_steps-1):										# fill the last entry of the history in phi at delay_steps-1
					# print( '\n\nhere is also STILL A PROBLEM HERE: if there is no perturbation, the history should grow constantly until delay_steps (included)\n')
					#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
					inst_Freq = (phi[idx_time,:]-phi[idx_time-1,:])/(dt*2*np.pi)	# calculate all instantaneous frequencies of all PLLs
					#print('instantaneous frequency when delta_perturbation is set: ', inst_Freq)
					#print('self.F_Omeg when perturbation is set: ', F_Omeg, '\n')
					#print('CHECK WHETHER CALCULATED AT THE TIME STEP; HERE FOR CALCULTED: instantaneous frequency TOWARDS last step of history:', inst_Freq)
					#print('number of oscis:', Nplls)
					cPD_t[idx_time]	 = [pll.pdc.cPD_t][0]
					if histtype == 'syncstate':
						phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, phiS[i], inst_Freq[i]) for i,pll in enumerate(pll_list)]
					elif histtype == 'uncoupled':
						phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, 0, inst_Freq[i]) for i,pll in enumerate(pll_list)]
					#print('new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
			''' NOW SIMULATE THE SYSTEM AFTER HISTORY IS SET '''
			for idx_time in range(delay_steps,Nrelax+delay_steps):
				''' adiabatic case, relaxation time '''
				#if idx_time == delay_steps:
				#	print('\n\nSIMULATION STARTS HERE, phase histories are set\n')
				#if idx_time < delay_steps+10:
				#	print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
				cPD_t[idx_time]	 = [pll.pdc.cPD_t][0]
				phi[idx_time+1,:] = [pll.nextrelax(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
				#if idx_time < delay_steps+10:
				#	print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
			for idx_time in range(Nrelax+delay_steps,Nsteps+Nrelax+delay_steps-1):
				''' adiabatic case, adiabatic change '''
				cPD_t[idx_time]	 = [pll.pdc.cPD_t][0]
				phi[idx_time+1,:] = [pll.nextadiab(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above

		elif c==0 and Kadiab_value_r>0 and cPD==0:
			''' This is for delay_steps > 0, cPD-rate > 0  '''
			Nrelax 		  = int(Trelax/dt)
			print('Relaxation time in [sec]:', Trelax)
			rate 		  = 0.1 / Trelax
			print('Rate of adiabatic change in [dim_of_Kadiabatic/sec]:', rate)
			rate_per_step = 0.1 / Nrelax
			''' use Trelax to calculate the rate and Nsteps until K_adiabatic is close to zero ...and back!'''
			Nsteps  	  = 2 * int( (2.0*np.pi*(Kadiab_value_r - K)) / rate_per_step )
			print('Simulation time after relaxation time in [s]:', Nsteps*dt, ' and total time in [s]:', Trelax+Nsteps*dt)
			phi = np.empty([Nsteps+Nrelax+delay_steps,Nplls])					# prepare container for phase time series
			Kadiab_t = np.empty([Nsteps+Nrelax+delay_steps])					# prepare container for c_LF
			''' SET INITIAL HISTORY AND PERTURBATION '''
			for idx_time in range(delay_steps):									# iterate over Nsteps from 0 to "Nsteps + delay_steps" -> of that "delay_steps+1" is history
				''' Important note: the container for the phase variables [phi] is indexed from 0 onwards, i.e., idx_time==delay_steps-1 corresponds to the entry in the container
					with index delay_steps-1, and hence the container then has "delay_steps" number of entries.
					Also note however, that below we always set idx_time+1, i.e., when idx_time==delay_steps-1, idx_time+1==delay_steps-1+1==delay_steps is set and the history is
					complete (delay_steps * dt written in real time).'''
				if idx_time <= (delay_steps-2):						# fill phi entries 1 to "delay_steps-2", note: we set idx_time+1 in the last call at idx_time==delay_steps-2
					#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
					Kadiab_t[idx_time]	 = [pll.vco.Kt][0]
					phi[idx_time+1,:] = [pll.setup_hist() for pll in pll_list]		# here the initial phase history is set
					#print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
				if idx_time == (delay_steps-1):										# fill the last entry of the history in phi at delay_steps-1
					# print( '\n\nhere is also STILL A PROBLEM HERE: if there is no perturbation, the history should grow constantly until delay_steps (included)\n')
					#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
					inst_Freq = (phi[idx_time,:]-phi[idx_time-1,:])/(dt*2*np.pi)	# calculate all instantaneous frequencies of all PLLs
					#print('instantaneous frequency when delta_perturbation is set: ', inst_Freq)
					#print('self.F_Omeg when perturbation is set: ', F_Omeg, '\n')
					#print('CHECK WHETHER CALCULATED AT THE TIME STEP; HERE FOR CALCULTED: instantaneous frequency TOWARDS last step of history:', inst_Freq)
					#print('number of oscis:', Nplls)
					Kadiab_t[idx_time]	 = [pll.vco.Kt][0]
					if histtype == 'syncstate':
						phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, phiS[i], inst_Freq[i]) for i,pll in enumerate(pll_list)]
					elif histtype == 'uncoupled':
						phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, 0, inst_Freq[i]) for i,pll in enumerate(pll_list)]
					#print('new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
			''' NOW SIMULATE THE SYSTEM AFTER HISTORY IS SET '''
			for idx_time in range(delay_steps,Nrelax+delay_steps):
				''' adiabatic case, relaxation time '''
				#if idx_time == delay_steps:
				#	print('\n\nSIMULATION STARTS HERE, phase histories are set\n')
				#if idx_time < delay_steps+10:
				#	print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
				Kadiab_t[idx_time]	 = [pll.vco.Kt][0]
				phi[idx_time+1,:] = [pll.nextrelaxKt(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
				#if idx_time < delay_steps+10:
				#	print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
			# print('transition from relaxation dynamics to adiabatic change at time: ', idx_time*dt, ' Kadiab_t[idx_time-1:idx_time+1]', Kadiab_t[(idx_time-1)],'    ', Kadiab_t[(idx_time)], '    ', Kadiab_t[(idx_time+1)],'\n')
			# print('return value from [pll.vco.Kt][0]: ', [pll.vco.Kt][0], ' at current time: ', idx_time*dt)
			for idx_time in range(Nrelax+delay_steps,Nsteps+Nrelax+delay_steps-1):
				''' adiabatic case, adiabatic change '''
				# print('return value from [pll.vco.Kt][0]: ', [pll.vco.Kt][0], ' at current time: ', idx_time*dt)
				# time.sleep(2)
				Kadiab_t[idx_time]	 = [pll.vco.Kt][0]
				phi[idx_time+1,:] = [pll.nextadiabKt(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
			Kadiab_t[idx_time+1]=Kadiab_t[idx_time]

	return {'phases': phi, 'intrinfreq': omega_0, 'coupling_strength': K_0, 'transdelays': delays_0, 'cPD': cPD_t, 'Kadiab_t': Kadiab_t}

''' CREATE PLL LIST '''
def generatePllObjects(mode,div,topology,couplingfct,histtype,Nplls,dt,c,delay,feedback_delay,F,F_Omeg,K,Fc,y0,phiM,domega,diffconstK,diffconstSendDelay,Nx,Ny,cPD,Trelax=0,Kadiab_value_r=0):

	cLF = 0;																	# low pass additive noise to the output signal of the PD, same effect like VCO noise according to model

	if not Nplls == Nx*Ny:
		print('Nplls was unequal to Nx*Ny, corrected for that, now Nplls=Nx*Ny.')
		Nplls = Nx*Ny

	if topology == 'global':
		G = nx.complete_graph(Nplls)
		# print('G and G.neighbors(PLL0):', G, G.neighbors(0)); sys.exit(1)

	elif ( topology == 'compareEntrVsMutual' and Nplls == 6):
		G = nx.DiGraph();
		G.add_nodes_from([i for i in range(Nplls)]);
		G.add_edges_from([(0,1),(1,0),(3,2)]);				  					# bidirectional coupling between 0 and 1 and 3 receives from 2, i.e., 2 entrains 3
		for i in range(Nplls):
			print('neighbors of oscillator ',i,':', list(G.neighbors(i)) , ' and egdes of',i,':', list(G.edges(i)))

	elif ( topology == 'entrainPLLsHierarch'):
		G = nx.DiGraph();
		level_hierarch = 9;
		if level_hierarch > Nplls:
			sys.exit('Special topology does not work like that... decrease hierarchy level - cannot exceed the number of PLLs in the system!')
		G.add_nodes_from([i for i in range(Nplls)]);
		for i in range(0,level_hierarch):
			G.add_edge(i+1 ,i);													# add unidirectional edge from osci 0 to 1, 1 to 2, and so on until level_hierarch is reached

		for i in range(level_hierarch+1, Nplls):
			G.add_edge(i, level_hierarch); 										# add unidirectional edge from highest hierarchy level to all remaining PLLS

		#for i in range(Nplls):
		#	print('neighbors of oscillator ',i,':', list(G.neighbors(i)) , ' and egdes of',i,':', list(G.edges(i)))
		# pos=nx.layout.spring_layout(G)
		# nx.draw_networkx_nodes(G, pos)
		# nx.draw_networkx_edges(G, pos, arrowstyle="->")
		# nx.draw_networkx_labels(G, pos)
		# plt.show()

	elif ( topology == 'ring' or topology == 'entrainAll' ):
		G = nx.cycle_graph(Nplls)
		# if topology == 'entrainAll':
		# 	G.remove_edge(0,2);
		# 	G.remove_edge(1,2);

	elif ( topology == 'chain' or topology == 'entrainOne' ):
		G = nx.path_graph(Nplls)
		# if topology == 'entrainOne':
		# 	G.remove_edge(0,2);
		# 	G.remove_edge(1,2);
		# 	G.remove_edge(2,1);

	# elif topology == 'entrainOne':
	# 	print('STOP, not working yet with G.neighbors(idx)! Topology of entrainment of synchronized state -- reference feeds into one of the oscillators.')
	# 	if not Nplls == 3:
	# 		print('Not yet configured for N != 3! Check.')
	# 	G = nx.MultiDiGraph();
	# 	G.add_edges_from([(0,1),(1,2),(2,1)]);
	#
	# elif topology == 'entrainAll':
	# 	print('STOP, not working yet with G.neighbors(idx)! Topology of entrainment of synchronized state -- reference feeds into all of the oscillators.')
	# 	if not Nplls == 3:
	# 		print('Not yet configured for N != 3! Check.')
	# 	G = nx.MultiDiGraph();
	# 	G.add_edges_from([(0,1),(0,2),(1,2),(2,1)]);

	else:
		N = np.sqrt(Nplls)
		if Nx == Ny:
			if N.is_integer():													# indirect check, whether N is an integer, which it should be for Nx=Ny as above checked
				N = int(N)
			else:
				raise ValueError('Npll is not valid: sqrt(N) is not an integer')

		if topology == 'square-open':
			G = nx.grid_2d_graph(Nx,Ny)

		elif topology == 'square-periodic':
			G = nx.grid_2d_graph(Nx,Ny, periodic=True)                            # for periodic boundary conditions:

		elif topology == 'hexagon':
			print('\nIf Nx =! Ny, then check the graph that is generated again!')
			G=nx.grid_2d_graph(Nx,Ny)											# why not ..._graph(Nx,Ny) ? NOTE the for n in G: loop has to be changed, loop over nx and ny respectively, etc....
			for n in G:
				x,y=n
				if x>0 and y>0:
					G.add_edge(n,(x-1,y-1))
				if x<Nx-1 and y<Ny-1:
					G.add_edge(n,(x+1,y+1))

		elif topology == 'hexagon-periodic':
			G=nx.grid_2d_graph(Nx,Ny, periodic=True)
			for n in G:
				x,y=n
				G.add_edge(n, ((x-1)%Nx, (y-1)%Ny))

		elif topology == 'octagon':												# why not ..._graph(Nx,Ny) ? NOTE the for n in G: loop has to be changed, loop over nx and ny respectively, etc....
			print('\nIf Nx =! Ny, then check the graph that is generated again!')
			G=nx.grid_2d_graph(Nx,Ny)
			for n in G:
				x,y=n
				if x>0 and y>0:
					G.add_edge(n,(x-1,y-1))
				if x<Nx-1 and y<Ny-1:
					G.add_edge(n,(x+1,y+1))
				if x<Nx-1 and y>0:
					G.add_edge(n,(x+1,y-1))
				if x<Nx-1 and y>0:
					G.add_edge(n,(x+1,y-1))
				if x>0 and y<Ny-1:
					G.add_edge(n,(x-1,y+1))

		elif topology == 'octagon-periodic':
			G=nx.grid_2d_graph(Nx,Ny, periodic=True)
			for n in G:
				x,y=n
				G.add_edge(n, ((x-1)%Nx, (y-1)%Ny))
				G.add_edge(n, ((x-1)%Nx, (y+1)%Ny))

		# G = nx.convert_node_labels_to_integers(G)
		G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted') # converts 2d coordinates to 1d index of integers, e.g., k=0,...,N-1

	#for i in range(Nplls):
	#	print('type G.neighbors(',i,')', list(G.neighbors(i)))
	# exit(0)

	# print('c=',c,' coupling-function:', couplingfct,'\n')
	# print('Complete this part for all cases, e.g. the case of K drawn from a distribution.')
	# if domega == 0:
	# 	print('Eigentfrequencies are identical, i.e., not distributed. domega=', domega)
	# else:
	# 	print('Eigentfrequencies are not identical, i.e., distributed with diffusion-constant domega=', domega)
	# if diffconstK == 0:
	# 	print('Coupling strengths are identical, i.e., not distributed. diffconstK=', diffconstK)
	# else:
	# 	print('Coupling strengths are not identical, i.e., distributed with diffusion constant diffconstK=', diffconstK)

	if cPD==0 and Trelax==0:
		if c==0:																# case of no dynamical noise (GWN)
			# print('Initiate (digital) PLL objects. Simulate without additive noise, triangular coupling function.')
			if couplingfct == 'sin':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									SinPhaseDetectComb(idx_pll, G.neighbors(idx_pll),div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'cos':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									CosPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'sincos':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									SinCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'triang':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
				# NOTE print('The coupling topology is given by:', G, ' for topology:', topology)

				# ''' Test and check '''
				# print('Container with initial perturbations, length, type, shape:', len(phiM), type(phiM), phiM.shape)
				# for idx_pll in range(Nplls):
				# 	print('index PLLs:',idx_pll, '    and G.neighbors(index_pll):', G.neighbors(idx_pll))
				#
				# import matplotlib
				# import matplotlib.pyplot as plt
				# fig, axs = plt.subplots(1,2)
				# labels=nx.draw_networkx_labels(G, pos=nx.spring_layout(G), ax=axs[0])
				# nx.draw(G, pos=nx.spring_layout(G), node_size=(0.5+phiM)*1000 ,ax=axs[0])
				# nx.draw(G, node_size=(0.5+phiM)*1000,ax=axs[1])
				# ''' Test and check '''

				if ( Nplls==3 and ( mode==2 or mode==1 or mode==0 ) and couplingfct == 'OFF' and not ( topology == 'entrainOne' or topology == 'entrainAll') ):
					print('\nSPECIAL MODE: individual intrinsic frequencies!\n')		#triang
					sys.exit('coupling function hacked! see simulation.py')
					F_intrin=[0.996, 1.004, 0.997];							    #[1.006, 1.008, 1.011];	[0.996, 1.004, 0.997]# put here the frequencies in Hz of the PLLs in the experimental setup_hist
					K_k     =[0.4135, 0.3995, 0.408]						    # the coupling strengths [0.3995, 0.3904, 0.3984] [0.4135, 0.3995, 0.408]
					Fc_k    =[0.015, 0.015, 0.015] 								#[0.04986, 0.0496, 0.049953] # the cut-off frequencies
					# F_intrin=[1.006, 1.008, 1.011];							# put here the frequencies in Hz of the PLLs in the experimental setup_hist
					# K_k     =[0.4045, 0.408, 0.4065]							# the coupling strengths
					# Fc_k    =[0.0154, 0.0154, 0.0154]							# the cut-off frequencies
					pll_list = [ PhaseLockedLoop(								# setup PLLs and storage in a list as PLL class objects
										Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
										PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
										LowPass(Fc_k[idx_pll],dt,K_k[idx_pll],F_Omeg,F_intrin[idx_pll],cLF,Trelax=0,y=y0),
										VoltageControlledOscillator(F_intrin[idx_pll],Fc_k[idx_pll],F_Omeg,K_k[idx_pll],dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
										)  for idx_pll in range(Nplls) ]		# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

				if ( Nplls==3 and ( mode==2 or mode==1 or mode==0 ) and couplingfct == 'triang'  and ( topology == 'entrainOne' or topology == 'entrainAll' or topology == 'feedforwardChain') ):
					print('\nSPECIAL MODE: individual intrinsic frequencies! (entrainment of self-org. sync. states if one K_k=0)\n')
					F_intrin=[0.996, 1.004, 0.8505];						    #[1.006, 1.008, 1.011];	# put here the frequencies in Hz of the PLLs in the experimental setup_hist
					K_k     =[0.408, 0.408, 0.0]							    	# the coupling strengths [0.3995, 0.3904, 0.3984] zero coupling strength for reference...
					Fc_k    =[0.015, 0.015, 0.015] 								#[0.04986, 0.0496, 0.049953] # the cut-off frequencies
					# F_intrin=[1.006, 1.008, 1.011];							# put here the frequencies in Hz of the PLLs in the experimental setup_hist
					# K_k     =[0.4045, 0.408, 0.4065]							# the coupling strengths
					# Fc_k    =[0.0154, 0.0154, 0.0154]							# the cut-off frequencies
					pll_list = [ PhaseLockedLoop(								# setup PLLs and storage in a list as PLL class objects
										Delayer(delay,dt,feedback_delay,diffconstSendDelay),	# delayer takes a time series and returns values at t and t-tau
										PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
										LowPass(Fc_k[idx_pll],dt,K_k[idx_pll],F_Omeg,F_intrin[idx_pll],cLF,Trelax=0,y=y0),
										VoltageControlledOscillator(F_intrin[idx_pll],Fc_k[idx_pll],F_Omeg,K_k[idx_pll],dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
										)  for idx_pll in range(Nplls) ]		# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
				else:
					print('\nNORMAL MODE triang mode! Homogeneous parameters.\n')
					pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
										Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
										PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
										LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
										VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
										)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'triangshift':									# triang(x) + a * triang(b*x)
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									PhaseDetectorCombinerShifted(idx_pll, G.neighbors(idx_pll), div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		else:
			if couplingfct == 'sin':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									SinPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'cos':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									CosPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'sincos':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									SinCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'triang':
				if ( Nplls==6 and ( mode==2 or mode==1 or mode==0 ) and topology == 'compareEntrVsMutual' ):
					print('\nSPECIAL MODE: 2 mutually delay-coupled PLLs, Ref entrains one PLL, free running PLL, free running Ref)\n')
					F_intrin=[1.0, 1.0, 1.02, 0.98, 1.0, 1.0];					#[PLLmut1, PLLmut2, Ref, PLLentrained, freePLL, freeRef];	# put here the frequencies in Hz of the PLLs in the experimental setup_hist
					K_k     =[0.4, 0.4, 0.4, 0.4, 0.4, 0.4];					# the coupling strengths -> set to zero for free running VCO, otherwise closed-loop PLL old:zero coupling strength for reference and free running
					Fc_k    =[0.015, 0.015, 0.015, 0.015, 0.015, 0.015];		# the cut-off frequencies
					c_k		=[1E-6, 1E-6, 1E-8, 1E-6, 1E-6, 1E-8];
					# F_intrin=[1.006, 1.008, 1.011];							# put here the frequencies in Hz of the PLLs in the experimental setup_hist
					# K_k     =[0.4045, 0.408, 0.4065]							# the coupling strengths
					# Fc_k    =[0.0154, 0.0154, 0.0154]							# the cut-off frequencies
					pll_list = [ PhaseLockedLoop(								# setup PLLs and storage in a list as PLL class objects
										Delayer(delay,dt,feedback_delay,diffconstSendDelay),	# delayer takes a time series and returns values at t and t-tau
										PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
										LowPass(Fc_k[idx_pll],dt,K_k[idx_pll],F_Omeg,F_intrin[idx_pll],cLF,Trelax=0,y=y0),
										NoisyVoltageControlledOscillator(F_intrin[idx_pll],Fc_k[idx_pll],F_Omeg,K_k[idx_pll],dt,domega,diffconstK,histtype,c_k[idx_pll],phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
										)  for idx_pll in range(Nplls) ]		# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
				elif ( Nplls==9 and ( mode==2 or mode==1 or mode==0 ) and couplingfct == 'triang' and topology == 'square-open' ):
					HF_components = False;
					K_k = np.zeros(Nplls)+K; c_k = np.zeros(Nplls)+c; F_k = np.zeros(Nplls)+F;
					if False:
						K_k[0] = 0; c_k[0] = c_k[0]*0.01; F_k[0] = 1.0;
						print('Set coupling strength PLL0 to %.2f and reduced its noise strength by %.2f.' %(K_k[0], c_k[0]/c_k[1]))
					if HF_components == True:
						print('\nSPECIAL MODE: N coupled PLLs, simulation including HF terms of the PD!\n')
						pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
										Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
										PhaseDetectorCombinerHighFreq(idx_pll, G.neighbors(idx_pll), div),
										LowPass(Fc,dt,K_k[idx_pll],F_Omeg,F_k[idx_pll],cLF,Trelax=0,y=y0),
										NoisyVoltageControlledOscillator(F_k[idx_pll],Fc,F_Omeg,K_k[idx_pll],dt,domega,diffconstK,histtype,c_k[idx_pll],phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
										)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
					else:
						print('\nSPECIAL MODE: N coupled PLLs, simulation DOES NOT include HF terms of the PD!\n')
						pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
										Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
										PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
										LowPass(Fc,dt,K_k[idx_pll],F_Omeg,F_k[idx_pll],cLF,Trelax=0,y=y0),
										NoisyVoltageControlledOscillator(F_k[idx_pll],Fc,F_Omeg,K_k[idx_pll],dt,domega,diffconstK,histtype,c_k[idx_pll],phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
										)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
				elif ( Nplls==9 and ( mode==2 or mode==1 or mode==0 ) and couplingfct == 'triang' and topology == 'entrainPLLsHierarch' ):
					print('\nSPECIAL MODE: one higher quality clock entrains via x hierarchy layers a set of PLLs!\n')
					c_k = np.zeros(Nplls)+c; F_k = np.zeros(Nplls)+F; K_k = np.zeros(Nplls)+K;
					c_k[0] = c_k[0]*0.01; F_k[0] = 1.0; #print(F_k)
					HF_components = False;
					print('Set entrainment with fref=%.2f hierarchy tree and reduced its noise strength by %.2f.' %(F_k[0], c_k[0]/c_k[1]))
					if HF_components == True:
						print('HF-coupling terms activated!')
						pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
											Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
											PhaseDetectorCombinerHighFreq(idx_pll, G.neighbors(idx_pll), div),
											LowPass(Fc,dt,K_k[idx_pll],F_Omeg,F_k[idx_pll],cLF,Trelax=0,y=y0),
											NoisyVoltageControlledOscillator(F_k[idx_pll],Fc,F_Omeg,K_k[idx_pll],dt,domega,diffconstK,histtype,c_k[idx_pll],phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
											)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
					else:
						print('HF-coupling terms DEactivated!')
						pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
											Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
											PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
											LowPass(Fc,dt,K_k[idx_pll],F_Omeg,F_k[idx_pll],cLF,Trelax=0,y=y0),
											NoisyVoltageControlledOscillator(F_k[idx_pll],Fc,F_Omeg,K_k[idx_pll],dt,domega,diffconstK,histtype,c_k[idx_pll],phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
											)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
				else:
					# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
					# print('phiM in setup PLLs:', phiM)
					pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
										Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
										PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
										LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
										NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
										)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'triangshift':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
				# print('phiM in setup PLLs:', phiM)
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									PhaseDetectorCombinerShifted(idx_pll, G.neighbors(idx_pll)),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
	elif cPD>0 and Trelax==0:
		if c==0:																# case of no dynamical noise (GWN)
			# print('Initiate (digital) PLL objects. Simulate without additive noise, triangular coupling function.')
			if couplingfct == 'sin':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(									# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),		# delayer takes a time series and returns values at t and t-tau
									NoisySinPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'cos':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisyCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'sincos':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisySinCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'triang':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisyPhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'triangshift':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisyPhaseDetectorCombinerShifted(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
		else:
			if couplingfct == 'sin':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisySinPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'cos':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisyCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'sincos':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisySinCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'triang':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisyPhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

			if couplingfct == 'triangshift':
				# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
				pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
									Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
									NoisyPhaseDetectorCombinerShifted(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
									LowPass(Fc,dt,K,F_Omeg,F,cLF,Trelax=0,y=y0),
									NoisyVoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
									)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

	elif cPD>0 and Trelax>0 and c==0 and Kadiab_value_r==0:
		if couplingfct == 'sin':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								NoisySinPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
								LowPassAdiabaticC(Fc,dt,K,F_Omeg,F,cLF,Trelax,y=y0),
								VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'cos':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								NoisyCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
								LowPassAdiabaticC(Fc,dt,K,F_Omeg,F,cLF,Trelax,y=y0),
								VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'sincos':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								NoisySinCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
								LowPassAdiabaticC(Fc,dt,K,F_Omeg,F,cLF,Trelax,y=y0),
								VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
		if couplingfct == 'triang':
			# print('\n\nTrelax:',Trelax,'\n\n')
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								NoisyPhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
								LowPassAdiabaticC(Fc,dt,K,F_Omeg,F,cLF,Trelax,y=y0),
								VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'triangshift':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								NoisyPhaseDetectorCombinerShifted(idx_pll, G.neighbors(idx_pll),dt,cPD,div),
								LowPassAdiabaticC(Fc,dt,K,F_Omeg,F,cLF,Trelax,y=y0),
								VoltageControlledOscillator(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

	elif cPD==0 and Trelax>0 and c>0 and Kadiab_value_r==0:
		if couplingfct == 'sin':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								SinPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeC(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'cos':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								CosPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeC(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'sincos':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								SinCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeC(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'triang':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeC(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'triangshift':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								PhaseDetectorCombinerShifted(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeC(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
	elif cPD==0 and Trelax>0 and c==0 and Kadiab_value_r>0:
		print('DO NOT USE THiS YET; FIRST CURE, introduced real cPD and cLF, fix, also add adiabatic case for PD noise')
		if couplingfct == 'sin':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								SinPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeK(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,Kadiab_value_r,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'cos':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								CosPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeK(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,Kadiab_value_r,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'sincos':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								SinCosPhaseDetectComb(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeK(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,Kadiab_value_r,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'triang':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeK(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,Kadiab_value_r,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'triangshift':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt,feedback_delay,diffconstSendDelay),			# delayer takes a time series and returns values at t and t-tau
								PhaseDetectorCombinerShifted(idx_pll, G.neighbors(idx_pll), div),
								LowPass(Fc,dt,K,F_Omeg,F,cPD,Trelax,y=y0),
								NoisyVoltageControlledOscillatorAdiabaticChangeK(F,Fc,F_Omeg,K,dt,domega,diffconstK,histtype,c,Trelax,Kadiab_value_r,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

	else:
		print('\n\nNOTE:   If Trelax>0, also cPD(initial) needs to be > zero. Noise diffusion const c should be zero! Fix that!\n\n');

	return pll_list
