# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10

@author: Adrian Wong
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
##  Glossary ##
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Reservoir Parameters
    N           -   dimension of reservoir / number of nodes in reservoir
    activ       -   activation function, defaults to np.tanh
    bias        -   adds a bias output node to the W fit (advised AGAINST using this)

    # Connection Matrix and Parameters
    A           -   adjacency/connection matrix with spectral radius rho
    rho         -   spectral radius
    density     -   percentage of non-zero nodes in A

    # Input Matrix and Parameters
    B           -   input weight matrix (unnormalized)
    sigma       -   scaling/normalization factor

    # Output Matrix and Parameters
    mask        -   to specify which index of the output should be used as input
    D           -   dimension of the input data
    W           -   output weight matrix 
    alpha       -   ridge parameter for least squares fit 
    
    # Continuous Time Parameters (diffRC only)
    ds          -   reservoir time step (constant or vector)
    integrator  -   time stepping method 
    
    # Listening Time Series
    M           -   number of steps of the incoming data
    r           -   reservoir state / internal representation (N x M)
    y_in        -   (external) input data (D x M) 
    y_target    -   (external) target/output data
    y_est       -   reservoir estimate when fitting r to y (D x M)
    
    # Echoing Time Series
    M_echo      -   number of steps to echo (value not stored explicitly in object)
    r_echo      -   state of the autonomous reservoir
    y_echo      -   prediction of the original dataset using the autonomous reservoir
    
    # Inference Time Series
    M_infer     -   number of steps to infer (value not stored explicitly in object)
    y_drive     -   (external) signal that continues to be presented to the reservoir 
    r_infer     -   reservoir state if the signal continues to be presented after training
    y_infer     -   state reconstruction assuming that signal continues to be presented
                    after training
                
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Dependencies ##
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
# import scipy as sp
from scipy.linalg import eigvals as eigvals
from scipy.sparse import random as sparseRandom
from scipy.optimize import fsolve as fsolve
from scipy.sparse import csr_matrix as sparseCsrMatrix
from scipy.sparse import diags as sparseDiags
from scipy.sparse.linalg import eigs as eigs
from scipy.sparse.linalg import svds as svds
from scipy.stats import uniform as statsUniform
from scipy.stats import norm as statsNormal
import time as time
import sys as sys
import os as os

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## Reservoir Class ##
The Reservoir class provides the overarching structure of the RC, where
a differential RC (diffRC) and a forward-map RC (mapRC) can be derived.
This class cannot be directly initiated, since the specific RC strucuture
is not yet specified. The step() function is missing from this class, but is 
specified in the subclasses.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
    def makeConnectionMat(self,rho,density=0.02,zeroDiag=False,dist=statsUniform,loc=-1.0,scale=2.0):
        # connection matrix is currently constructed based on sparsity, but will be updated to be based on degree in the future
        # dist options: stats.uniform or stats.normal
        # default is uniform in range [-1,1), otherwise [loc,loc+scale)
        # normal distribution is N(loc,scale)
        self.rho = rho
        self.density = density
        self.A = sparseRandom(self.N, self.N, density = self.density, data_rvs = dist(loc,scale).rvs)
        # zero diagonal option
        if zeroDiag:
            self.A.setdiag(0)
            self.A.eliminate_zeros()
        # find spectral radius, i.e. Largest Magnitude (LM) eigenvalue
        maxEig = np.abs(eigs(self.A, k = 1, which='LM', return_eigenvectors=False))
        # maxEig = svds(self.A, k = 1, which='LM', return_singular_vectors=False)
        # rescale to specified spectral radius
        self.A = self.A.multiply(self.rho/maxEig)
        print("Connection matrix is setup.")
        
    def makeDiagConnectionMat(self,rho=1,randMin=-1.0,randMax=1.0):
        # default is uniform in range [-1,1)
        self.rho = rho
        diagElem = np.random.uniform(randMin,randMax,self.N)
        # rescale to specified spectral radius
        self.A = sparseDiags(diagElem).multiply(self.rho/max(diagElem))
        print("Diagonal-Only Connection matrix is setup.")

    def makeInputMat(self,D,sigma,randMin=0.0,randMax=1.0,sparseFlag=True):
        # dist options: np.random.uniform or np.random.normal
        # default is uniform in range [0,1), otherwise [randMin,randMax)
        # this uniform function is different from the one used in makeConnectionMat()
        self.D = D
        self.sigma = sigma
        # either sparse or full option
        if sparseFlag:
            val = self.sigma*np.random.uniform(low=randMin,high=randMax,size=self.N)
            self.B = sparseCsrMatrix((val, (np.arange(self.N), np.sort(np.arange(self.N)%self.D))), shape=(self.N, self.D))
        else:
            self.B = np.random.random(size=(self.N,self.D))
            for i in range(self.N):
                self.B[i] = self.sigma*self.B[i]/np.linalg.norm(self.B[i])   
        print("Input matrix is setup.")
        
    # def makeInputMat(self,sigma,randMin=-1.0,randMax=1.0,sparseFlag=True):
    #     # dist options: np.random.uniform or np.random.normal
    #     # default is uniform in range [0,1), otherwise [randMin,randMax)
    #     # this uniform function is different from the one used in makeConnectionMat()
    #     self.sigma = sigma
    #     # either sparse or full option
    #     if sparseFlag:
    #         row = np.arange(self.N)
    #         col = np.sort(np.argmax(np.random.rand(self.N, self.D), axis = 1)) # chooses one of D inputs
    #         val = self.sigma*np.random.uniform(low=randMin,high=randMax,size=self.N)
    #         self.B = sparseCsrMatrix((val, (row, col)), shape=(self.N, self.D))
    #     else:
    #         self.B = np.random.random(size=(self.N,self.D))
    #         for i in range(self.N):
    #             self.B[i] = self.sigma*self.B[i]/np.linalg.norm(self.B[i])   
    #     print("Input matrix is setup.")

# Listening Functions
    def listen(self,y_in,randFlag=False,randMin=-1.5,randMax=1.5):
        # check for exception
        # if len(y.shape[0]) != self.D:
        #     raise Exception('Shape of input data y(t) should be in the shape of DxM.')
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
        self.M = y_in.shape[1]
        for i in range(1,self.M):
            self.r[:,i] = self.step(self.r[:,i-1],y_in[:,i-1])
            self.progressBar(i,self.M)
        # if len(y_in.shape)>1:
        #     self.M = y_in.shape[1]
        #     for i in range(1,self.M):
        #         self.r[:,i] = self.step(self.r[:,i-1],y_in[:,i-1])
        #         self.progressBar(i,self.M)
        # else:
        #     self.M = len(y_in)
        #     for i in range(1,self.M):
        #         self.r[:,i] = self.step(self.r[:,i-1],y_in[i-1])
        #         self.progressBar(i,self.M)
        # wall time
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
        if mask == None:
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
        if self.bias:
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

    # def infer2(self,y_drive,driveIndex,measInterval,randFlag=False,randMin=-10,randMax=10):
    #     # check for exception
    #     if y_drive.shape[0]!=len(driveIndex):
    #         raise Exception('Shape of driving data does not match the number of drive variables.')
    #     # starting timer
    #     print("Inference 2 phase in progress...")
    #     startTime = time.time()
    #     # establish r_infer
    #     M_infer = y_drive.shape[1]
    #     self.infer2Setup(M_infer,randFlag,randMin,randMax)
    #     # inferring main loop
    #     for i in range(1,M_infer):
    #         # placeholder vector
    #         yTemp = self.W@self.r_infer2[:,i-1]
    #         self.r_infer2[:self.N,i] = self.step(self.r_infer2[:self.N,i-1],yTemp)
    #         if i%measInterval==0:
    #             # replacing measured variables with data
    #             yTemp[driveIndex] = np.copy(y_drive[:,i-1])
    #             self.r_infer2[:self.N,i] = self.step(self.r_infer2[:self.N,i-1],yTemp)
    #         self.progressBar(i,M_infer)
    #     self.y_infer2 = self.W@self.r_infer2
    #     # wall time
    #     print(f"\nInference 2 phase completed. Time taken: {time.time()-startTime:.3} seconds.")
    #     print('-----------------------------------------------------------------')

    # def inferImplicit(self,y_drive,driveIndex,measInterval,randFlag=False,randMin=-10,randMax=10):
    #     # check for exception
    #     if y_drive.shape[0]!=len(driveIndex):
    #         raise Exception('Shape of driving data does not match the number of drive variables.')
    #     # starting timer
    #     print("Inference Implicit phase in progress...")
    #     startTime = time.time()
    #     # establish r_infer
    #     M_infer = y_drive.shape[1]
    #     self.inferImplicitSetup(M_infer,randFlag,randMin,randMax)
    #     # inferring main loop
    #     rTemp = self.r_inferImplicit[:,0]
    #     for i in range(1,M_infer):
    #         # placeholder vector
    #         yTemp = self.W@self.r_inferImplicit[:,i-1]
    #         self.r_inferImplicit[:self.N,i] = self.step(self.r_inferImplicit[:self.N,i-1],yTemp)
    #         if i%measInterval==0:
    #             # replacing measured variables with data
    #             self.r_inferImplicit[:self.N,i] = fsolve(self.consistencyImplicit,self.r_inferImplicit[:self.N,i],
    #             args=(y_drive[:,i],driveIndex))
    #         self.progressBar(i,M_infer)
    #     self.y_inferImplicit = self.W@self.r_inferImplicit
    #     # wall time
    #     print(f"\nInference 2 phase completed. Time taken: {time.time()-startTime:.3} seconds.")
    #     print('-----------------------------------------------------------------')

    # def infer2Setup(self,M_infer,randFlag,randMin,randMax):
    #     self.r_infer2 = np.ones((self.r.shape[0],M_infer))
    #     self.r_infer2[:self.N,0] = self.step(self.r[:self.N,-1],self.W@self.r[:,-1])
    #     # randFlag for testing sychronization
    #     if randFlag == True:
    #         print("Perturbing Reservoir States...")
    #         self.r_infer2[:self.N,0] += np.random.uniform(low=randMin,high=randMax,size=self.N)

    # def inferImplicitSetup(self,M_infer,randFlag,randMin,randMax):
    #     self.r_inferImplicit = np.ones((self.r.shape[0],M_infer))
    #     self.r_inferImplicit[:self.N,0] = self.step(self.r[:self.N,-1],self.W@self.r[:,-1])
    #     # randFlag for testing sychronization
    #     if randFlag == True:
    #         print("Perturbing Reservoir States...")
    #         self.r_inferImplicit[:self.N,0] += np.random.uniform(low=randMin,high=randMax,size=self.N)
            
    # # Miscellaneous Function
    # def consistencyImplicit(self,rTemp,rTemp2,yTemp):
    #     return rTemp2-self.step(rTemp,self.W@rTemp)

    # def consistencyImplicit(self,rTemp,y_drive,driveIndex):
    #     if self.bias:
    #         yTemp = self.W@np.hstack((rTemp,1))
    #         yTemp[driveIndex] = np.copy(y_drive)
    #         return self.step(rTemp[:self.N],yTemp)-self.step(rTemp[:self.N],self.W@np.hstack((rTemp,1)))
    #     else:
    #         yTemp = self.W@rTemp
    #         yTemp[driveIndex] = np.copy(y_drive)
    #         return self.step(rTemp,yTemp)-self.step(rTemp,self.W@rTemp)

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## mapRC subclass ##
The mapRC subclass inherits the Reservoir class structure.
It is defined by a forward map, which gives it a simpler structure.
The mapRC class has its own step() function.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class mapRC(Reservoir):
    def __init__(self,N,activ=np.tanh,bias=False):
        super().__init__(N,activ=activ,bias=bias)
        print('-----------------------------------------------------------------')
        print("Forward Map Reservoir initiated.")

    def reservoirForwardMap(self,r,y):
        return self.activ(self.A@r+self.B@y)

    def step(self,r,y):
        return self.reservoirForwardMap(r,y)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## diffRC subclass ##
The diffRC subclass inherits the Reservoir class structure.
It is defined by a vector field, hence an integrator needs 
to be specified. For now the choice of integrators are 
'RK2' and 'RK4', and a reservoir timestep (ds) must also
be specified, in addition to the usual parameters.
The diffRC class has its own step() function.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class diffRC(Reservoir):
    def __init__(self,N,ds,activ=np.tanh,bias=False):
        self.ds = ds
        self.integrator = self.RK2
        super().__init__(N,activ=activ,bias=bias)
        print('-----------------------------------------------------------------')
        print("Differential Reservoir initiated.")

    def chooseIntegrator(self,integratorName='RK2'):
        if integratorName == 'RK2':
            self.integrator = self.RK2
        elif integratorName == 'RK4':
            self.integrator = self.RK4
        else:
            print("Invalid integrator. Choose 'RK2' or 'RK4'.")

    def reservoirVectorField(self,r,y):
        return self.ds*(-r+self.activ(self.A@r+self.B@y))

    def step(self,r,y):
        return self.integrator(r,y,self.reservoirVectorField)

# Explicit Integrators for differential Reservoir
    def RK4(self,r,y,f):
        k1 = f(r     , y)
        k2 = f(r+k1/2, y)
        k3 = f(r+k2/2, y)
        k4 = f(r+k3  , y)
        return r + (k1+2*k2+2*k3+k4)/6  

    def RK2(self,r,y,f):
        k1 = f(r     , y)
        k2 = f(r+k1/2, y)
        return r + k2