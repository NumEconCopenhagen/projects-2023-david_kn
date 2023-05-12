from types import SimpleNamespace
import time
import numpy as np

class EndogeneousGrowthModelClass():
    
    def __init__(self,do_print=True):

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        print('calling .setup()')
        print('The model simulates from period 0 to T')
        self.setup(simTotal = input('Choose T: '))

        print('calling .allocate()')
        self.allocate()

    def setup(self, simTotal):
        """ baseline parameters """

        par = self.par

        par.s = 0.5
        par.n = 0.1
        par.alpha = 0.5
        par.delta = 0.2
        par.phi = 0.5

        par.K_lag_ini = 1.0 # initial capital stock
        par.L_lag_ini = 1.0 # initial labor stock
        par.simT = int(simTotal) + 1 # length of simulation
        
    def allocate(self):
        """ allocate arrays for simulation """

        par = self.par
        sim = self.sim

        # a. list of variables
        vars = ['K','L','A','S','Y']
        prices = ['w','r']

        # b. allocate
        allvarnames = vars + prices
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)

    def simulate(self,do_print=True):
        """ simulate model """

        par = self.par
        sim = self.sim

        t0 = time.time()
        
        # a. initial values
        sim.K[0] = par.K_lag_ini
        sim.L[0] = par.L_lag_ini

        # b. iterate
        for t in range(par.simT):

            simulate_(par,sim,t)

            if t < par.simT - 1:
                simulate_acc(par,sim,t)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def simulate_(par,sim,t):

    # Technology
    sim.A[t] = sim.K[t]**par.phi

    # Production function
    sim.Y[t] = sim.K[t]**par.alpha * (sim.A[t]*sim.L[t])**(1-par.alpha)

    # Factor prices
    sim.r[t] = par.alpha * sim.K[t]**(par.alpha-1) * (sim.A[t]*sim.L[t])**(1-par.alpha)
    sim.w[t] = (1-par.alpha) * sim.K[t]**(par.alpha) * (sim.A[t]*sim.L[t])**(-par.alpha) * sim.A[t]

    # Savings 
    sim.S[t] = par.s * sim.Y[t]

def simulate_acc(par,sim,t):        

    # Capital accumulation
    sim.K[t+1] = sim.S[t] + (1-par.delta)*sim.K[t]

    # Labor accumulation
    sim.L[t+1] = (1+par.n)*sim.L[t]