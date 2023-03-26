
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self, alpha = 0.5, sigma = 1):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = alpha
        par.sigma = sigma

        # d. wages
        par.wM = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF,WF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = WF*LM + WF*LF

        # b. home production
        if par.sigma == 0:
            H = np.minimum(HM, HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha) * HM ** ((par.sigma-1)/par.sigma) + par.alpha * HF ** ((par.sigma-1)/par.sigma)) ** (par.sigma / (par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self, WF = 1.0, do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF,WF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve_continuous(self, WF = 1.0, do_print=False):
        """ solve model continously """

        opt = SimpleNamespace()

        def value_of_choice(x):

            return -self.calc_utility(x[0],x[1],x[2],x[3],WF)
        
        # Set bounds 
        bounds = ((0,24),(0,24),(0,24),(0,24))
        
        # call solver, use SLSQP
        initial_guess = [1,2,1,1]

        solver = optimize.minimize(value_of_choice, initial_guess, bounds=bounds)

        # unpack solution
        opt.LM = solver.x[0]
        opt.HM = solver.x[1]
        opt.LF = solver.x[2]
        opt.HF = solver.x[3]

        # print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
        
        return opt

    def solve_wF_vec(self, discrete=False,  do_print=False):
        """ solve model for vector of female wages """
       
        par = self.par
        sol = self.sol

        for i, WF in enumerate(par.wF_vec):

            if discrete == True:
            
                opt = self.solve_discrete(WF)
            
            else:

                opt = self.solve_continuous(WF)
        
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF

        opt = np.array([sol.LM_vec, sol.HM_vec, sol.LF_vec, sol.HF_vec])

        return opt
        

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass

    def return_par(self):

        par = self.par
        x = par.rho

        return x

        
