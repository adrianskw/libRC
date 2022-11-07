#%% -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:56:30 2022

@author: Adrian Wong
"""
%reload_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
import sys as sys
sys.path.append('../')
from libRC import diffRC,mapRC

plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = [15, 3]

def RK4(y,t,dt,f,params):
    k1 = dt*f(y,t,params)
    k2 = dt*f(y+k1/2, t+dt/2,params)
    k3 = dt*f(y+k2/2, t+dt/2,params)
    k4 = dt*f(y+k3, t+dt,params)
    return y + (k1+2*k2+2*k3+k4)/6
    
def burnIn(Mburn,x0,t,dt,model,params):
    for i in range(1,Mburn):
        x0=RK4(x0,t,dt,model,params)
    return x0

def forwardInt(M,x,t,dt,model,params):
    for i in range(1,M):
        x[:,i]=RK4(x[:,i-1],t,dt,model,params)
    return x

def lorenz63(u,t,params):
    [sigma,rho,beta] = params
    [x, y, z] = u
    dxdt = sigma*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y - beta*z
    return np.asarray([dxdt,dydt,dzdt])


#%% generate data
np.random.seed(11111)
D = 3
N = 150

M = 10000
Mpred = 10000
Mplot = 4000
dt = 0.01
t = 0 # dummy variable

sigma = 10   # Prandlt number
rho = 28     # Rayleigh number
beta = 8.0/3
params = [sigma,rho,beta]

x = np.zeros((D,M))
xPred = np.zeros((D,Mpred))
x[:,0] = [12,13,14]
x[:,0] = burnIn(3456,x[:,0],t,dt,lorenz63,params)
x = forwardInt(M,x,t,dt,lorenz63,params)
y = x+np.random.normal(loc=0.0,scale=0.5,size=x.shape)

xPred[:,0] = RK4(x[:,-1],t,dt,lorenz63,params)
xPred = forwardInt(Mpred,xPred,t,dt,lorenz63,params)
yPred = xPred+np.random.normal(loc=0.0,scale=1,size=xPred.shape)
#%% standard RC usage
np.random.seed(11111)
ds = 0.2
RC = diffRC(N,D,ds,bias=True)
RC.chooseIntegrator('RK2')

rho = 0.9
sigma = 0.1
RC.makeConnectionMat(rho,loc=-1,scale=2)
RC.makeInputMat(sigma,randMin=0,randMax=1)

RC.listen(y)
RC.train(y,alpha=0.001)
RC.echo(Mpred)

driveIndex =[1]
RC.infer(yPred[driveIndex],driveIndex)

plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = [15, 3]
for i in range(D):
    plt.plot(yPred[i,:Mplot],label='data')
    plt.plot(RC.yEcho[i,:Mplot],label='echo')
    plt.plot(RC.yInfer[i,:Mplot],label='infer')
    plt.legend(loc='upper right')
    plt.show()


plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(xPred[0],xPred[1],'.')
plt.plot(RC.yEcho[0],RC.yEcho[1])
plt.show()

# %%
Anew = np.zeros((N+1,N+1))
Bnew = np.zeros((N+1,D  ))
# Anew = np.zeros((N,N))
# Bnew = np.zeros((N,D  ))
Anew[:N,:N] = RC.A.todense()
Bnew[:N,:D] = RC.B.todense()
Cnew = Anew+Bnew@RC.W
Dnew = Cnew-np.eye(Cnew.shape[0])
print('SR(A)\t\t', sp.linalg.eigvals(Anew).max())
print('SR(B@W)\t\t', sp.linalg.eigvals(Bnew@RC.W).max())
print('SR(A+B@W)\t', sp.linalg.eigvals(Cnew).max())
print('SR(-1+A+B@W)\t', sp.linalg.eigvals(Dnew).max())
print('mu(A+B@W)\t',sp.linalg.eigvals((Cnew.T+Cnew)/2).max())
print('mu(-1+A+B@W)\t',sp.linalg.eigvals((Dnew.T+Dnew)/2).max())
print('mu(-1+A)\t',sp.linalg.eigvals((Anew.T+Anew)/2).max())
# %%
