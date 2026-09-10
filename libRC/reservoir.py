# -*- coding: utf-8 -*-
"""Reservoir base class and the mapRC/diffRC subclasses.

The Reservoir class provides the overarching structure of the RC, where
a differential RC (diffRC) and a forward-map RC (mapRC) can be derived.
This class cannot be directly initiated, since the specific RC structure
is not yet specified. The step() function is missing from this class, but
is specified in the subclasses.
"""
import sys
import time

import numpy as np
from scipy.linalg import eigvals
from scipy.stats import uniform as statsUniform

from . import connectivity
from .integrators import RK2, RK4


# Reservoir class SHOULD NOT BE DIRECTLY CALLABLE
# Initialize with diffRC or mapRC subclasses instead
class Reservoir():
    def __init__(self,N,activ=np.tanh,bias=False):
        # basic RC parameters
        self.N          = N
        self.activ      = activ
        self.bias       = bias # better for mapRC, better without for diffRC
        # parameters for dynamics
        self.A          = None
        self.rho        = None
        self.density    = None
        self.degree     = None
        # parameters for driver
        self.B          = None
        self.sigma      = None
        # parameters for fit
        self.mask       = None
        self.D          = None # created in makeInputMat
        self.W          = None
        self.alpha      = None
        # reservoir trajectories
        self.M          = None
        self.r          = None
        self.y_est      = None
        # prediction/echo trajectories
        self.r_echo     = None
        self.y_echo     = None

# Setup Functions for Matrices
    def makeConnectionMatDegree(self,rho,degree=3,diag_vals=None,dist=statsUniform,loc=-1.0,scale=2.0,rng=None,tol=0,maxiter=None):
        # dist options: stats.uniform or stats.normal
        # default is uniform in range [-1,1), otherwise [loc,loc+scale)
        # normal distribution is N(loc,scale)
        self.rho = rho
        self.A,self.degree = connectivity.makeConnectionMatDegree(
            self.N,rho,degree=degree,diag_vals=diag_vals,dist=dist,loc=loc,scale=scale,
            rng=rng,tol=tol,maxiter=maxiter)
        print("Connection matrix is setup.")

    def makeConnectionMatDensity(self,rho,density=0.02,diag_vals=None,dist=statsUniform,loc=-1.0,scale=2.0,rng=None,tol=0,maxiter=None):
        self.rho = rho
        self.density = density
        self.A = connectivity.makeConnectionMatDensity(
            self.N,rho,density=density,diag_vals=diag_vals,dist=dist,loc=loc,scale=scale,
            rng=rng,tol=tol,maxiter=maxiter)
        print("Connection matrix is setup.")

    def makeDiagConnectionMat(self,rho=1,randMin=-1.0,randMax=1.0,rng=None):
        # default is uniform in range [-1,1)
        self.rho = rho
        self.A = connectivity.makeDiagConnectionMat(self.N,rho=rho,randMin=randMin,randMax=randMax,rng=rng)
        print("Diagonal-Only Connection matrix is setup.")

    def makeInputMat(self,D,sigma,randMin=0.0,randMax=1.0,sparseFlag=True,rng=None):
        # dist options: np.random.uniform or np.random.normal
        # default is uniform in range [0,1), otherwise [randMin,randMax)
        # this uniform function is different from the one used in makeConnectionMat()
        self.D = D
        self.sigma = sigma
        self.B = connectivity.makeInputMat(self.N,D,sigma,randMin=randMin,randMax=randMax,sparseFlag=sparseFlag,rng=rng)
        print("Input matrix is setup.")

# Listening Functions
    def listen(self,y_in,randFlag=False,randMin=-1.5,randMax=1.5):
        # starting timer
        print("Listening phase in progress...")
        startTime = time.time()
        # establish r
        if len(y_in.shape)>1:
            self.M = y_in.shape[1]
        else:
            self.M = len(y_in)
        #setup
        self.listenSetup(randFlag,randMin,randMax)
        # listening main loop
        for i in range(1,self.M):
            self.r[:,i] = self.step(self.r[:,i-1],y_in[:,i-1])
            self.progressBar(i,self.M)
        print(f"\nListening phase completed. Time taken: {time.time()-startTime:.3} seconds.")
        print('-----------------------------------------------------------------')

    def listenSetup(self,randFlag,randMin,randMax):
        # randFlag for testing sychronization
        if randFlag:
            print("Perturbing Reservoir States...")
            self.r = np.random.uniform(low=randMin,high=randMax,size=(self.N,self.M))
        else:
            self.r = np.zeros((self.N,self.M))

# Training with Linear Fit
    def train(self,y_target,start=0,end=None,alpha=0.02,mask=None):
        if mask is None:
            if y_target.shape[0] != self.D:
                print("Mismatch in input vs target dimensions. Mask required. Halting code.")
                return -1
            self.mask = np.eye(self.D)
        else:
            self.mask = mask
        # small hack to make default 'end' from class variables
        if end is None:
            end = self.M
        # adding bias term
        if self.bias and self.r.shape[0] == self.N:
            self.r = np.vstack([self.r,np.ones(self.M)])
        # alpha is the regularization parameter of ridge regression
        self.alpha = alpha

        # starting timer
        startTime = time.time()
        print("Training in progress...")
        # run fit to solve for W (refer to the README)
        RRT = self.r[:,start:end]@self.r[:,start:end].T + self.alpha * np.eye(self.r.shape[0])
        URT = y_target[:,start:end]@self.r[:,start:end].T
        self.W = np.linalg.solve(RRT,URT.T).T # RRT is symmetric so RRT = RRT.T
        # determining reconstructed state
        self.y_est = np.zeros(y_target.shape)
        self.y_est[:,start:end] = self.W@self.r[:,start:end]
        # determining fit error and wall time
        self.fitError = np.sqrt(np.linalg.norm(self.y_est-y_target)**2)/self.M/self.D
        print(f'Fit Error: {self.fitError:12.4f}')
        print(f"Training phase completed. Time taken: {time.time()-startTime:.3} seconds.")
        print('-----------------------------------------------------------------')

# Echoing Functions
    def echo(self,M_echo,randFlag=False,randMin=-10,randMax=10):
        # starting timer
        print("Echoing phase in progress...")
        startTime = time.time()
        # establish r_echo
        self.echoSetup(M_echo,randFlag,randMin,randMax)
        # echoing main loop
        for i in range(1,M_echo):
            self.r_echo[:self.N,i] = self.step(self.r_echo[:self.N,i-1],self.mask@self.W@self.r_echo[:,i-1])
            self.progressBar(i,M_echo)
        self.y_echo = self.W@self.r_echo
        # wall time
        print(f"\nEchoing phase completed. Time taken: {time.time()-startTime:.3} seconds.")
        print('-----------------------------------------------------------------')

    def echoSetup(self,M_echo,randFlag,randMin,randMax):
        self.r_echo = np.ones((self.r.shape[0],M_echo))
        self.r_echo[:self.N,0] = self.step(self.r[:self.N,-1],self.mask@self.W@self.r[:,-1])
        # randFlag for testing sychronization
        if randFlag == True:
            print("Perturbing Reservoir States...")
            self.r_echo[:self.N,0] += np.random.uniform(low=randMin,high=randMax,size=self.N)

# Inference Functions
    def infer(self,y_drive,driveIndex,randFlag=False,randMin=-10,randMax=10):
        # check for exception
        if y_drive.shape[0]!=len(driveIndex):
            raise Exception('Shape of driving data does not match the number of drive variables.')
        # starting timer
        print("Inference phase in progress...")
        startTime = time.time()
        # establish r_infer
        M_infer = y_drive.shape[1]
        self.inferSetup(M_infer,randFlag,randMin,randMax)
        # inferring main loop
        for i in range(1,M_infer):
            # placeholder vector
            yTemp = self.mask@self.W@self.r_infer[:,i-1]
            # replacing measured variables with data
            yTemp[driveIndex] = np.copy(y_drive[:,i-1])
            self.r_infer[:self.N,i] = self.step(self.r_infer[:self.N,i-1],yTemp)
            self.progressBar(i,M_infer)
        self.y_infer = self.mask@self.W@self.r_infer
        # wall time
        print(f"\nInference phase completed. Time taken: {time.time()-startTime:.3} seconds.")
        print('-----------------------------------------------------------------')

    def inferSetup(self,M_infer,randFlag,randMin,randMax):
        self.r_infer = np.ones((self.r.shape[0],M_infer))
        self.r_infer[:self.N,0] = self.step(self.r[:self.N,-1],self.mask@self.W@self.r[:,-1])
        # randFlag for testing sychronization
        if randFlag == True:
            print("Perturbing Reservoir States...")
            self.r_infer[:self.N,0] += np.random.uniform(low=randMin,high=randMax,size=self.N)

    def progressBar(self,i,N):
        if i%62 == 0:
            sys.stdout.write(f"\r{100*(i+1)/N:8.1f}%")
            sys.stdout.flush()
        if i+1 == N:
            sys.stdout.write(f"\r{100*(i+1)/N:8.1f}%")
            sys.stdout.flush()

    def calcEchoSR(self):
        if self.bias:
            Anew = np.zeros((self.N+1,self.N+1))
            Bnew = np.zeros((self.N+1,self.D  ))
        else:
            Anew = np.zeros((self.N  ,self.N  ))
            Bnew = np.zeros((self.N  ,self.D  ))
        Anew[:self.N,:self.N] = self.A.todense()
        Bnew[:self.N,:self.D] = self.B.todense()
        print(eigvals(Anew+Bnew@self.W-2*np.eye(Anew.shape[0])).max())
        print(eigvals(Anew+Bnew@self.W).max())

    def inferPC(self,y,start=0,end=-1):
        D,M = y.shape
        yy = y - np.swapaxes([np.mean(y,axis=1)],0,1)
        y_infer = self.y_infer -np.swapaxes([np.mean(self.y_infer,axis=1)],0,1)
        return (y_infer@yy.T)**2/((y_infer@y_infer.T)*(yy@yy.T))


class mapRC(Reservoir):
    """The mapRC subclass inherits the Reservoir class structure. It is
    defined by a forward map, which gives it a simpler structure. The
    mapRC class has its own step() function."""
    def __init__(self,N,activ=np.tanh,bias=False):
        super().__init__(N,activ=activ,bias=bias)
        print('-----------------------------------------------------------------')
        print("Forward Map Reservoir initiated.")

    def reservoirForwardMap(self,r,y):
        return self.activ(self.A@r+self.B@y)

    def step(self,r,y):
        return self.reservoirForwardMap(r,y)


class diffRC(Reservoir):
    """The diffRC subclass inherits the Reservoir class structure. It is
    defined by a vector field, hence an integrator needs to be specified.
    For now the choice of integrators are 'RK2' and 'RK4', and a
    reservoir timestep (ds) must also be specified, in addition to the
    usual parameters. The diffRC class has its own step() function."""
    def __init__(self,N,ds,activ=np.tanh,bias=False):
        self.ds = ds
        self.integrator = RK2
        super().__init__(N,activ=activ,bias=bias)
        print('-----------------------------------------------------------------')
        print("Differential Reservoir initiated.")

    def chooseIntegrator(self,integratorName='RK2'):
        if integratorName == 'RK2':
            self.integrator = RK2
        elif integratorName == 'RK4':
            self.integrator = RK4
        else:
            print("Invalid integrator. Choose 'RK2' or 'RK4'.")

    def reservoirVectorField(self,r,y):
        return self.ds*(-r+self.activ(self.A@r+self.B@y))

    def step(self,r,y):
        return self.integrator(r,y,self.reservoirVectorField)
