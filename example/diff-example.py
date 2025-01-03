#%% -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:56:30 2022

@author: Adrian Wong
"""
# %reload_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from sys import path as path
path.append('../')
from libRC import diffRC,mapRC
plt.rcParams.update({'font.size': 16})

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
N = 100

M = 15000
Mpred = M
Mplot = M//2
dt = 0.01
t = 0 # dummy variable

sigma = 10   # Prandlt number
rho = 28     # Rayleigh number
beta = 8.0/3
params = [sigma,rho,beta]

noiseLevel = 0.5
x = np.zeros((D,M))
xPred = np.zeros((D,Mpred))
x[:,0] = [12,13,14]
x[:,0] = burnIn(3456,x[:,0],t,dt,lorenz63,params)
x = forwardInt(M,x,t,dt,lorenz63,params)
y = x+np.random.normal(loc=0.0,scale=noiseLevel,size=x.shape)

xPred[:,0] = RK4(x[:,-1],t,dt,lorenz63,params)
xPred = forwardInt(Mpred,xPred,t,dt,lorenz63,params)
yPred = xPred+np.random.normal(loc=0.0,scale=noiseLevel,size=xPred.shape)
#%% standard RC usage#%% standard RC usage
np.random.seed(11111)
ds = 0.2
rho = 0.9
sigma = 2/max(np.max(y,1)-np.min(y,1))/ds

RC = diffRC(N,D,ds,bias=False)

RC.makeConnectionMat(rho,density=0.02,loc=-1,scale=1)
RC.makeInputMat(sigma,randMin=-1,randMax=1)

RC.listen(y)
RC.train(y,alpha=0.01)
RC.echo(Mpred)

driveIndex =[1]

RC.infer(yPred[driveIndex],driveIndex)
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = [15, 10]
fig,axs = plt.subplots(3,sharex='col')
fig.subplots_adjust(hspace=0)
labelList = ['x','y','z']
for i in range(D):
    axs[i].plot(yPred[i,:Mplot],label='data')
    axs[i].plot(RC.yEcho[i,:Mplot],label='echo')
    axs[i].plot(RC.yInfer[i,:Mplot],label='infer')
    axs[i].set_xlabel('steps')
    axs[i].set_yticks([])
    axs[i].set_ylabel(labelList[i])
axs[0].legend(loc='upper right')
axs[0].set_title('Prediction Window')
plt.show()


plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(xPred[0],xPred[1],label='data')
plt.plot(RC.yEcho[0],RC.yEcho[1],label='echo')
plt.plot(RC.yInfer[0],RC.yInfer[1],label='infer')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Attractors')
plt.legend()
plt.show()

PCmat = RC.inferPC(yPred)
plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(yPred[0],RC.yInfer[0],'.')
plt.plot(yPred[1],RC.yInfer[1],'.')
plt.xlabel('data')
plt.ylabel('infer')
plt.title(f'Reconstruction PC = {PCmat[0,0]:0.3f}')
plt.legend(['Unobserved','Observed'])
plt.show()
# %%
