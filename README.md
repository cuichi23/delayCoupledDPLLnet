# delayCoupledDPLLnet

simulate DPLL(first-order-LF)-networks for different connection topologies, distributed intrinsic frequencies, distributed coupling strength, [distributed transmission delays], dynamic freq. noise and statistical analysis many noisy realizations; also contains a 'bruteforce'  analyzer of the basins of attraction for m-twist synchronized states

##############################################################################################################################################################

HOWTO USE:
**********

A: Prepare the setup file: /[programs]/coupled_DPLLs/1params.txt
set multiprocessing mode, discretization for bruteforce mode, number of child-processes for multiproc., the coupling function, intrinsic freqency, 
sample frequency, diffusion constant of GWN distributed intrinsic frequencies, diffusion constant of GWN distribution of the coupling strengths

B: Run the Program
bash: goto the local directory of the program and 
cd /home/[user]/.../coupled_DPLLs

python case_[sim_mode].py [topology] [#osci] [K] [F_c] [delay] [F_Omeg] [k] [Tsim] [c] [Nsim] [N entries for the value of the perturbation to oscis]

where: 
	[sim_mode] 
		'singleout', one realization, 
		'bruteforce', basin of attraction about spezified state
		'noisy', simulate many realizations of a network with dyn. freq noise, optional distributed PLL component parameters
		'oracle', mode to use with design centering algorithm Josefine
	[topology]
		'string': choose from {'global','ring','chain','square-open', 'square-periodic','hexagon','octagon'}

	[#osci]
		'Integer', for 2D grids use integers N whose square root yields an integer value 
	[K]
		'float', (mean) coupling strength VCO, related to VCO sensitivity
	[F_c]
		'float', (mean) cut-off frequency of the filters
	[delay]
		'float', (mean) transmisson delay
	[F_Omeg]
		'float', frequency of the synchronized state for the given set of parameters Omega = omega + K * h( -Omega tau )
	[k]
		'integer in {0, ... N-1}', N is the # of oscis, specifies which twist solution is used, where k=0 implies the in phase synched 								   state; also {-N/2,...,0,...,N/2} with the resulting restrictions
	[Tsim]
		'unsigned integer', simulation time in multiples of the eigenperiod 	
	[c]
		'float', diffusion constant of the GWN freq. noise process - \sqrt(2*D) = std, var = 2*D
	[Nsim]
		'unsigned integer': number of realizations - in singleout and oracle this is set to Nsim=1, independent of 							    the command input
	[perturbation]
		'list of floats, separated by spaces': an initial delta perturbation added to the histories of the DPLL of the network at time 									       t-dt
	

EXMPLES:

python case_singleout.py ring 3 0.25 0.5 0.53 1.143846576 0 75 0 1 0. 0. 0.

simulates a single realization in a [ring] topology with N=[3] oscis, coupling strength K=[0.25] Hz, cut-off frequency fc=[0.5] Hz, delay tau=[0.53], global frequency of the (determinist) expected state FOmeg=[1.143846576] Hz, m-twist number k=[0], simulation length in Tsim=[75], diffusion constant of the dynamical noise c=[0], number of realizations Nsim=[1], and the three delta perturbations that could be added to the N=3 oscis at the end of the history at t=-dt, here given by three zeros and hence not adding any perturbation

