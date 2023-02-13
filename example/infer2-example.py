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

M = 50000
Mpred = 4200
Mplot = 4200
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
N = 100
rho = 0.9
sigma = 0.1

mode = 'map'
# mode = 'diff'
if mode == 'map':
    RC = mapRC(N,D,bias=True)
elif mode == 'diff':
    ds = 0.3
    RC = diffRC(N,D,ds,bias=True)
    RC.chooseIntegrator('RK2')

RC.makeConnectionMat(rho,loc=-1,scale=2)
RC.makeInputMat(sigma,randMin=0,randMax=1)

RC.listen(y)
RC.train(y,alpha=0.01)
RC.echo(Mpred)

driveIndex =[1]
measInterval = 15
RC.infer(yPred[driveIndex],driveIndex)
RC.infer2(yPred[driveIndex],driveIndex,measInterval)

plt.rcParams['figure.figsize'] = [15, 4]
plt.plot(yPred[0,:Mplot],'.',label='data')
# plt.plot(RC.yEcho[0,:Mplot],label='echo')
plt.plot(RC.yInfer[0,:Mplot],label='infer')
plt.plot(RC.yInfer2[0,:Mplot],label='infer2')
plt.legend()
plt.show()

ticks = measInterval*(np.arange(int(Mpred/measInterval)))
plt.semilogy(np.abs(xPred[0,:Mplot]-RC.yInfer2[0,:Mplot]),label='infer2 error')
plt.semilogy(ticks,np.abs(xPred[0,:Mplot:measInterval]-RC.yInfer2[0,:Mplot:measInterval]),'.')
plt.show()

#%%
plt.rcParams['figure.figsize'] = [15, 10]
fig,axs = plt.subplots(3,sharex='col')
fig.subplots_adjust(hspace=0)
labelList = ['x','y','z']
for i in range(D):
    axs[i].plot(yPred[i,:Mplot],'.',label='data')
    # axs[i].plot(RC.yEcho[i,:Mplot],label='echo')
    axs[i].plot(RC.yInfer[i,:Mplot],label='infer')
    axs[i].plot(RC.yInfer2[i,:Mplot],label='infer2')
    axs[i].set_xlabel('steps')
    axs[i].set_yticks([])
    axs[i].set_ylabel(labelList[i])
axs[0].legend(loc='upper right')
axs[0].set_title('Prediction Window')
plt.show()

plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(xPred[0],xPred[1],'.',label='data')
# plt.plot(RC.yEcho[0],RC.yEcho[1],label='echo')
plt.plot(RC.yInfer[0],RC.yInfer[1],label='infer')
plt.plot(RC.yInfer2[0],RC.yInfer2[1],label='infer2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Attractors')
plt.legend()
plt.show()

PCmat = RC.inferPC(yPred)
plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(yPred[0],RC.yInfer2[0],',',label='infer2')
plt.plot(yPred[0],RC.yInfer[0],',',label='infer')
plt.xlabel('data')
plt.ylabel('infer')
plt.legend()
plt.title(f'Reconstruction PC = {PCmat[0,0]:0.3f}')
plt.show()

# %%
