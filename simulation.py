#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import numpy as np
import networkx as nx
from scipy.signal import sawtooth

''' CLASSES

authors: Alexandros Pollakis, Lucas Wetzel [emails: lwetzel@pks.mpg.de]
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
		x_ctrl = self.lf.next(x_comb)											# the filtering at the loop filter is applied to the phase detector signal
		phi = self.vco.next(x_ctrl)[0]											# the control signal is used to update the VCO and thereby evolving the phase
		return phi

	def setup_hist(self):														# set the initial history of the PLLs, all evolving with frequency F_Omeg, provided on program start
		phi = self.vco.set_initial()[0]											# [0] vector-entry zero of the two values saved to phi which is returned
		return phi

	def set_delta_pertubation(self,idx_time,phi,phiS,inst_Freq):
		x_ctrl = self.lf.set_initial_control_signal(phi,idx_time,inst_Freq)		# the filtering at the loop filter is applied to the phase detector signal
		#print('x =', x, ', x_delayed =', x_delayed, ', x_comb =', x_comb, ', x_ctrl =', x_ctrl)
		phi = self.vco.delta_perturbation(phi,phiS,x_ctrl)[0]
		return phi

	def next_magic(self):														# UNCLEAR: how is that different from next_free? below... lf.y does not change during the time that next_magic is called
		x_ctrl = self.lf.y														# set control signal; here: lf.y = 0 at the beginning
		phi = self.vco.next(x_ctrl)[0]											# [0] vector-entry zero of the two values returned
		return phi

	def next_free(self):														# evolve phase of PLLs as if they were uncoupled, hence x_ctrl = 0.0
		phi = self.vco.next(0.0)[0]
		return phi

# LF: y = integral du x(t) * p(t-u)												# this can be expressed in terms of a differential equation with the help of the Laplace transform
class LowPass:
	"""A lowpass filter class"""
	def __init__(self,Fc,dt,K,F_Omeg,F,y=0,y_old=0):
		self.Fc = Fc															# set cut-off frequency
		self.F = F																# intrinsic frequency of VCO - here needed for x_k^C(0)
		self.F_Omeg = F_Omeg													# provide freq. of synchronized state under investigation - here needed for x_k^C(0)
		# here should be the value of K drawn for the VCO from the gaussian distribution, however it might also suffice to use the mean,
		# since the history is given as the perfectly synched state... that does not imply that the K become the same however!
		self.K_Hz = K
		self.K = 2.0*np.pi*K													# provide coupling strength - here needed for x_k^C(0)
		self.dt = dt															# set time-step
		self.beta = (dt*Fc) / (dt*Fc + 1)
		self.w_c = 2*np.pi*Fc													# angular cut-off frequency of the loop filter for a=1, filter of first order
		self.y = y																# denotes the control signal, output of the LF

	def set_initial_control_signal(self,phi,idx_time,inst_Freq):				# set the control signal for the last time step of the history, in the case the history is a synched state
		self.inst_Freq = inst_Freq												# calculate the instantaneous frequency for the last time step of the history
		#self.y = (self.F_Omeg - self.F) / (self.K)								# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
		if self.K!=0:															# this if-call is fine, since it will only be evaluated once
			self.y = (self.inst_Freq - self.F) / (self.K_Hz)					# calculate the state of the LF at the last time step of the history, it is needed for the simulation of the network
			# self.y = (2.0 * np.pi * (self.inst_Freq - self.F)) / (self.K)
		else:
			self.y = 0.0
		return self.y

	def next(self,x):							      							# this updates y=x_k^{C}(t), the control signal, using the input x=x_k^{PD}(t), the phase-detector signal
		x_ctrl0 = 0.0
		self.y = (1-self.beta)*self.y + self.beta*(x-x_ctrl0/self.Fc)			# the difference to the old version is the non-zero initial condition x_k^C(0)=[\dot{\theta}_k(0)-\omega] / K
		#print('state of filter AFTER update:', self.y)
		return self.y

# VCO: d_phi / d_t = omega + K * x
class VoltageControlledOscillator:
	"""A voltage controlled oscillator class"""
	def __init__(self,F,F_Omeg,K,dt,domega,diffconstK,c=None,phi=None):
		self.sOmeg = 2.0*np.pi*F_Omeg											# set angular frequency of synchronized state under investigation (sOmeg)
		self.diffconstK = diffconstK
		self.domega = domega
		if domega != 0.0:
			self.F = np.random.normal(loc=F, scale=np.sqrt(2.0*domega))			# set intrinsic frequency of the VCO plus gaussian dist. random variable from a distribution
			self.omega = 2.0*np.pi*self.F										# set intrinsic angular frequency of the VCO plus gaussian dist. random variable from a distribution
			# print('Intrinsic freq. from gaussian dist.:', self.omega, 'for diffusion constant domega:', self.domega)
		else:
			self.omega = 2.0*np.pi*F											# set intrinsic frequency of the VCO
		if diffconstK != 0:														# set input sensitivity of VCO [ok to do here, since this is only called when the PLL objects are created]
			self.K = np.random.normal(loc=K, scale=np.sqrt(2.0*diffconstK))		# provide coupling strength - here needed for x_k^C(0)
			self.K = 2.0*np.pi*self.K
			# print('2*pi*K from gaussian dist.:', self.K, 'for diffusion constant diffconstK:', self.diffconstK)
		else:
			self.K = 2.0*np.pi*K

		self.dt = dt															# set time step with which the equations are evolved
		self.phi = phi															# this is the internal representation of phi, NOT the container in simulateNetwork
		self.c = c																# noise strength -- chose something like variance or std here!

	def next(self,x_ctrl):														# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
		self.d_phi = self.omega + self.K * x_ctrl
		self.phi = self.phi + self.d_phi * self.dt
		return self.phi, self.d_phi

	def delta_perturbation(self, phi, phiS, x_ctrl):							# sets a delta-like perturbation 0-dt, the last time-step of the history
		self.d_phi = phiS + ( self.omega + self.K * x_ctrl ) * self.dt			# ADDITIVE DELTA PERTURBATION!
		self.phi = self.phi + self.d_phi
		return self.phi, self.d_phi

	def set_initial(self):														# sets the phase history of the VCO with the frequency of the synchronized state under investigation
		self.d_phi = self.sOmeg * self.dt
		self.phi = self.phi + self.d_phi
		return self.phi, self.d_phi

# + noise
class NoisyVoltageControlledOscillator(VoltageControlledOscillator):			# make a child class of VoltageControlledOscillator (inherits all functions of VoltageControlledOscillator)
	"""A voltage controlled oscillator class WITH noise (GWN)"""
	def next(self,x_ctrl):														# compute change of phase per time-step due to intrinsic frequency and noise (if non-zero variance)
																				# watch the separation of terms for order dt and the noise with order sqrt(dt)
		# self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + ( 2.0 * np.pi ) * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c*self.F)) * np.sqrt(self.dt) # scales with self.F
		self.d_phi = ( self.omega + self.K * x_ctrl ) * self.dt + ( 2.0 * np.pi ) * np.random.normal(loc=0.0, scale=np.sqrt(2.0*self.c)) * np.sqrt(self.dt) # no scaling of noise with frequency
		self.phi = self.phi + self.d_phi
		return self.phi, self.d_phi

	def delta_perturbation(self, phi, phiS, x_ctrl):							# sets a delta-like perturbation 0-dt, the last time-step of the history
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
	def __init__(self,idx_self,idx_neighbours):
		# print('Phasedetector and Combiner: sawtooth')
		self.h = lambda x: sawtooth(x,width=0.5)								# set the type of coupling function, here a sawtooth since we consider digital PLLs (rectangular signals)
		self.idx_self = idx_self												# assigns the index
		self.idx_neighbours = idx_neighbours									# assigns the neighbors according to the coupling topology

	def next(self,x,x_delayed,idx_time=0):										# gets time-series results at delayed time and current time to calculate phase differences
		try:
			x_self = x[self.idx_self]											# extract own state (phase) at time t and save to x_self
			#if idx_time == 59:
			#	print('x_self:', x_self)
			x_neighbours = x_delayed[self.idx_neighbours]						# extract the states of all coupling neighbors at t-tau and save to x_neighbours
			#if idx_time == 59:
			#	print('x_neighbours:', x_neighbours)
			self.y = np.mean( self.h( x_neighbours - x_self ) )					# calculate phase detector output signals and combine them to yield the signal that is fed into the loop filter
			#if idx_time == 59:
			#	print('phase detector signal:', self.y)
			return self.y
		except:																	# if there is no input (uncoupled PLL?), then the phase detector output is zero
			return 0.0

# y = 1 / n * sum h( x_delayed_neighbours - x_self )
class SinPhaseDetectComb(PhaseDetectorCombiner):								# child class for different coupling function - here cosinusoidal
	def __init__(self,idx_self,idx_neighbours):
		# print('Phasedetector and Combiner: sin(x)')
		self.h = lambda x: np.sin(x)											# set the type of coupling function, here a sine-function
		self.idx_self = idx_self												# assigns the index
		self.idx_neighbours = idx_neighbours									# assigns the neighbors according to the coupling topology

# y = 1 / n * sum h( x_delayed_neighbours - x_self )
class CosPhaseDetectComb(PhaseDetectorCombiner):									# child class for different coupling function - here sinusoidal
	def __init__(self,idx_self,idx_neighbours):
		# print('Phasedetector and Combiner: cos(x)')
		self.h = lambda x: np.cos(x)											# set the type of coupling function, here a sine-function
		self.idx_self = idx_self												# assigns the index
		self.idx_neighbours = idx_neighbours									# assigns the neighbors according to the coupling topology

# delayer
class Delayer:
	"""A delayer class"""
	def __init__(self,delay,dt):
		# print('Delayer set to identical transmission delays')
		self. delay = delay
		self.delay_steps = int(round(delay/dt))									# when initialized, the delay in time-steps is set to delay_steps
		#print('\ndelay steps:', self.delay_steps, '\n')

	def next(self,idx_time,x):
		idx_delayed = idx_time - self.delay_steps								# delayed index is calculated
		#if(idx_time >= (self.delay_steps) and idx_time < (self.delay_steps+2) ):
		#	print('\nidx_delayed', idx_delayed, 'with phi[t]=', x[idx_time,:],'with phi[t-tau]=', x[idx_delayed,:],'at idx_time', idx_time, '\n')
		#print('phases at time t:', np.asarray(x[idx_time,:]), 'phases at time t-tau:', np.asarray(x[idx_delayed,:]))
		return np.asarray(x[idx_time,:]), np.asarray(x[idx_delayed,:])			# x is is the time-series from which the values at t-dt and t-tau are returned

class DistDelayDelayer(Delayer):
	"""A delayer class"""
	def __init__(self,delay,dt,std_dist_delay,std_dyn_delay_noise):
		print('Delayer: FOR THIS CASE YOU HAVE TO FIND A SOLUTION FOR THE FREQUENCIES AND HISTORIES! -- take the mean delay to approximate the global frequency, also return distribution of delays')
		if std_dist_delay != 0:
			self.delay = np.random.normal(loc=delay, scale=std_dist_delay)		# process variation, the delays in the network are gaussian distributed about the mean delay
		else:
			self.delay = delay
																				# NOTE: static distribution of transmission delays - t
		self.delay_steps = int(round(self.delay/dt))							# when initialized, the delay in time-steps is set to delay_steps
		#print('\ndelay steps:', self.delay_steps, '\n')

	def next(self,idx_time,x):
		idx_delayed = idx_time - self.delay_steps								# delayed index is calculated
		#if(idx_time >= (self.delay_steps) and idx_time < (self.delay_steps+2) ):
		#	print('\nidx_delayed', idx_delayed, 'with phi[t]=', x[idx_time,:],'with phi[t-tau]=', x[idx_delayed,:],'at idx_time', idx_time, '\n')
		#print('phases at time t:', np.asarray(x[idx_time,:]), 'phases at time t-tau:', np.asarray(x[idx_delayed,:]))
		return np.asarray(x[idx_time,:]), np.asarray(x[idx_delayed,:])			# x is is the time-series from which the values at t-dt and t-tau are returned

class DistDelayDelayerWithDynNoise(Delayer):
	"""A delayer class"""
	def __init__(self,delay,dt,std_dist_delay,std_dyn_delay_noise):
		print('Delayer: FOR THIS CASE YOU HAVE TO FIND A SOLUTION FOR THE FREQUENCIES AND HISTORIES! -- take the mean delay to approximate the global frequency, also return distribution of delays')
		if std_dist_delay != 0:
			self.delay = np.random.normal(loc=delay, scale=std_dist_delay)		# process variation, the delays in the network are gaussian distributed about the mean delay
		else:																	# NOTE: static distribution of transmission delays - t
			self.delay = delay
		self.delay_steps = int(round(self.delay/dt))							# when initialized, the delay in time-steps is set to delay_steps
		#print('\ndelay steps:', self.delay_steps, '\n')

	def next(self,idx_time,x,std_dist_delay,std_dyn_delay_noise):
		idx_delayed = idx_time - self.delay_steps - int(round(np.random.normal(0.0, scale=std_dyn_delay_noise)/dt)) # delayed index is calculated
		#if(idx_time >= (self.delay_steps) and idx_time < (self.delay_steps+2) ):
		#	print('\nidx_delayed', idx_delayed, 'with phi[t]=', x[idx_time,:],'with phi[t-tau]=', x[idx_delayed,:],'at idx_time', idx_time, '\n')
		#print('phases at time t:', np.asarray(x[idx_time,:]), 'phases at time t-tau:', np.asarray(x[idx_delayed,:]))
		return np.asarray(x[idx_time,:]), np.asarray(x[idx_delayed,:])			# x is is the time-series from which the values at t-dt and t-tau are returned

''' SIMULATE NETWORK '''
def simulateNetwork(mode,Nplls,F,F_Omeg,K,Fc,delay,dt,c,Nsteps,topology,couplingfct,phiS,phiM,domega,diffconstK):
	y0 = 0																		# inital filter status:
	''' for the last step of the initial history, the filter status has to be set if one uses the second order (inertia) type description of the model;
 		this avoids performing the integration of the filter and reduces the dependence on an entiry history to a ODE of first order for the control signal;
		it is important however, to get the details associated to this transformation, concerning initial condition for the condtrol signal in the case of the ODE first order
		or instead the continuous history in case of the integration '''
	np.random.seed()															# restart pseudo random-number generator
	pll_list = generatePllObjects(mode,topology,couplingfct,Nplls,dt,c,delay,F,F_Omeg,K,Fc,y0,phiM,domega,diffconstK)	# create object lists of PLL objects of the network

	delay_steps = pll_list[0].delayer.delay_steps       						# get the number of steps representing the delay at a given time-step from delayer of PLL_0
	phi = np.empty([Nsteps+delay_steps,Nplls])									# prepare container for phase time series
	# here the initial phases of all PLLs in pll_list are copied into the first  entry of the container phi for the phases of the PLLs
	phi[0,:] = [pll.vco.phi for pll in pll_list]
	# print('phi[0,:] ->', phi[0,:])
	omega_0  = [pll.vco.omega for pll in pll_list]								# obtain the randomly distributed (gaussian) values for the intrinsic frequencies
	K_0      = [pll.vco.K for pll in pll_list]									# obtain the randomly distributed (gaussian) values for the coupling strength of the VCO
	delays_0 = [pll.delayer.delay for pll in pll_list]							# obtain the randomly distributed (gaussian) values for the transmission delays
	# print('omega_0 ->', omega_0)
	# this is the for-loop that iterates the system, first the initial conditions is set, then the dynamics are computed
	for idx_time in range(Nsteps+delay_steps-1):									# iterate over Nsteps from 0 to "Nsteps + delay_steps" -> of that "delay_steps+1" is history
		''' Important note: the container for the phase variables [phi] is indexed from 0 onwards, i.e., idx_time==delay_steps-1 corresponds to the entry in the container
		 	with index delay_steps-1, and hence the container then has "delay_steps" number of entries.
			Also note however, that below we always set idx_time+1, i.e., when idx_time==delay_steps-1, idx_time+1==delay_steps-1+1==delay_steps is set and the history is
			complete (delay_steps * dt written in real time).'''
		if idx_time <= (delay_steps-2):						# fill phi entries 1 to "delay_steps-2", note: we set idx_time+1 in the last call at idx_time==delay_steps-2
			#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
			phi[idx_time+1,:] = [pll.setup_hist() for pll in pll_list]			# here the initial phase history is set
			#print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
		if idx_time == (delay_steps-1):											# fill the last entry of the history in phi at delay_steps-1
			# print( '\n\nhere is also STILL A PROBLEM HERE: if there is no perturbation, the history should grow constantly until delay_steps (included)\n')
			#print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
			inst_Freq = (phi[idx_time,:]-phi[idx_time-1,:])/(dt*2*np.pi)		# calculate all instantaneous frequencies of all PLLs
			#print('instantaneous frequency when delta_perturbation is set: ', inst_Freq)
			#print('self.F_Omeg when perturbation is set: ', F_Omeg, '\n')
			#print('CHECK WHETHER CALCULATED AT THE TIME STEP; HERE FOR CALCULTED: instantaneous frequency TOWARDS last step of history:', inst_Freq)
			#print('number of oscis:', Nplls)
			phi[idx_time+1,:] = [pll.set_delta_pertubation(idx_time, phi, phiS[i], inst_Freq[i]) for i,pll in enumerate(pll_list)]
			#print('new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])
		if idx_time > (delay_steps-1):
			#if idx_time == delay_steps:
			#	print('\n\nSIMULATION STARTS HERE, phase histories are set\n')
			#if idx_time < delay_steps+10:
			#	print( 'prior [step =', idx_time ,'] entry phi-container simulateNetwork-fct:', phi[idx_time][:])
			phi[idx_time+1,:] = [pll.next(idx_time,phi) for pll in pll_list]	# now the network is iterated, starting at t=0 with the history as prepared above
			#if idx_time < delay_steps+10:
			#	print( 'new   [step =', idx_time+1 ,'] entry phi-container simulateNetwork-fct:', phi[idx_time+1][:], 'difference: ', phi[idx_time+1][:] - phi[idx_time][:])

	return {'phases': phi, 'intrinfreq': omega_0, 'coupling_strength': K_0, 'transdelays': delays_0}

''' CREATE PLL LIST '''
def generatePllObjects(mode,topology,couplingfct,Nplls,dt,c,delay,F,F_Omeg,K,Fc,y0,phiM,domega,diffconstK):
	if topology == 'global':
		G = nx.complete_graph(Nplls)
	elif topology == 'ring':
		G = nx.cycle_graph(Nplls)
	elif topology == 'chain':
		G = nx.path_graph(Nplls)
	else:
		N = np.sqrt(Nplls)
		if N.is_integer():
			N = int(N)
		else:
			raise ValueError('Npll is not valid: sqrt(N) is not an integer')
		if topology == 'square':
			G=nx.grid_2d_graph(N,N)
		elif topology == 'hexagon':
			G=nx.grid_2d_graph(N,N)
			for n in G:
				x,y=n
				if x>0 and y>0:
					G.add_edge(n,(x-1,y-1))
				if x<N-1 and y<N-1:
					G.add_edge(n,(x+1,y+1))
		elif topology == 'octagon':
			G=nx.grid_2d_graph(N,N)
			for n in G:
				x,y=n
				if x>0 and y>0:
					G.add_edge(n,(x-1,y-1))
				if x<N-1 and y<N-1:
					G.add_edge(n,(x+1,y+1))
				if x<N-1 and y>0:
					G.add_edge(n,(x+1,y-1))
				if x<N-1 and y>0:
					G.add_edge(n,(x+1,y-1))
				if x>0 and y<N-1:
					G.add_edge(n,(x-1,y+1))
		G = nx.convert_node_labels_to_integers(G)

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
	if c==0:																	# case of no dynamical noise (GWN)
		# print('Initiate (digital) PLL objects. Simulate without additive noise, triangular coupling function.')
		if couplingfct == 'sin':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt),								# delayer takes a time series and returns values at t and t-tau
								SinPhaseDetectComb(idx_pll, G.neighbors(idx_pll)),
								LowPass(Fc,dt,K,F_Omeg,F,y=y0),
								VoltageControlledOscillator(F,F_Omeg,K,dt,domega,diffconstK,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'cos':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt),								# delayer takes a time series and returns values at t and t-tau
								CosPhaseDetectComb(idx_pll, G.neighbors(idx_pll)),
								LowPass(Fc,dt,K,F_Omeg,F,y=y0),
								VoltageControlledOscillator(F,F_Omeg,K,dt,domega,diffconstK,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
		if couplingfct == 'triang':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt),								# delayer takes a time series and returns values at t and t-tau
								PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll)),
								LowPass(Fc,dt,K,F_Omeg,F,y=y0),
								VoltageControlledOscillator(F,F_Omeg,K,dt,domega,diffconstK,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
	else:
		if couplingfct == 'sin':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, sinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt),								# delayer takes a time series and returns values at t and t-tau
								SinPhaseDetectComb(idx_pll, G.neighbors(idx_pll)),
								LowPass(Fc,dt,K,F_Omeg,F,y=y0),
								NoisyVoltageControlledOscillator(F,F_Omeg,K,dt,domega,diffconstK,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

		if couplingfct == 'cos':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, cosinusoidal coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt),								# delayer takes a time series and returns values at t and t-tau
								CosPhaseDetectComb(idx_pll, G.neighbors(idx_pll)),
								LowPass(Fc,dt,K,F_Omeg,F,y=y0),
								NoisyVoltageControlledOscillator(F,F_Omeg,K,dt,domega,diffconstK,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]			# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided
		if couplingfct == 'triang':
			# print('Initiate (phase shifted) PLL objects. Simulate with additive noise, triangular coupling function.')
			pll_list = [ PhaseLockedLoop(										# setup PLLs and storage in a list as PLL class objects
								Delayer(delay,dt),								# delayer takes a time series and returns values at t and t-tau
								PhaseDetectorCombiner(idx_pll, G.neighbors(idx_pll)),
								LowPass(Fc,dt,K,F_Omeg,F,y=y0),
								NoisyVoltageControlledOscillator(F,F_Omeg,K,dt,domega,diffconstK,c,phi=phiM[idx_pll]) # set intrinsic frequency of VCO, frequency of synchronized state under investigation, coupling strength
								)  for idx_pll in range(Nplls) ]				# time-step value, and provide phiM, the phases at the beginning of the history that need to be provided

	return pll_list
