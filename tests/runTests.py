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
N = 50

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
#%%
np.random.seed(11111)
N = 500
ds = 0.1
RC = diffRC(N,D,ds,activ=np.cos,bias=True)
RC.chooseIntegrator('RK4')

rho = 0.9
sigma = 0.1
RC.makeConnectionMat(rho,density=0.03,zeroDiag=False,loc=0,scale=1)
RC.makeInputMat(sigma,randMin=-1,randMax=1)

RC.listen(y)
RC.train(y,start=50,end=M-1,alpha=0.01)
RC.echo(Mpred)

driveIndex =[1]
RC.infer(yPred[driveIndex],driveIndex)

cur1 = {'yHat':RC.yHat,'yEcho':RC.yEcho,'yInfer':RC.yInfer}
ref1 = np.load('test1.npz')

#%%
np.random.seed(11111)
N = 50
ds = 0.2
RC = diffRC(N,D,ds)
RC.chooseIntegrator('RK2')

rho = 0.9
sigma = 0.1
RC.makeConnectionMat(rho,density=0.03,zeroDiag=False,loc=-1,scale=1)
RC.makeInputMat(sigma,randMin=0,randMax=1)

RC.listen(y)
RC.train(y,alpha=0.001)
RC.echo(Mpred)

driveIndex =[1]
RC.infer(yPred[driveIndex],driveIndex)

cur2 = {'yHat':RC.yHat,'yEcho':RC.yEcho,'yInfer':RC.yInfer}
ref2 = np.load('test2.npz')

#%%

np.random.seed(11111)
N = 200
ds = 0.1
RC = diffRC(N,D,ds)
RC.chooseIntegrator('RK2')

rho = 0.8
sigma = 0.1
RC.makeConnectionMat(rho,density=0.05,zeroDiag=True,loc=-1,scale=2)
RC.makeInputMat(sigma,randMin=0,randMax=1)

RC.listen(y)
RC.train(y,alpha=0.01)
RC.echo(Mpred)

driveIndex =[1]
RC.infer(yPred[driveIndex],driveIndex)


cur3 = {'yHat':RC.yHat,'yEcho':RC.yEcho,'yInfer':RC.yInfer}
ref3 = np.load('test3.npz')

#%%
np.random.seed(11111)
N = 66
ds = 0.2
RC = diffRC(N,D,ds)
RC.chooseIntegrator('RK2')

rho = 0.8
sigma = 0.1
RC.makeConnectionMat(rho,density=0.02,zeroDiag=False,loc=-1,scale=2)
RC.makeInputMat(sigma,randMin=0,randMax=1)

RC.listen(y,randFlag=True,randMin=-1,randMax=-1)
RC.train(y,alpha=0.00)
RC.echo(Mpred,randFlag=True,randMin=-1,randMax=-1)

driveIndex =[1]
RC.infer(yPred[driveIndex],driveIndex,randFlag=True,randMin=-1,randMax=-1)

cur4 = {'yHat':RC.yHat,'yEcho':RC.yEcho,'yInfer':RC.yInfer}
ref4 = np.load('test4.npz')

# %%

print("---------------------------------------------------------------")
tol = {'yHat':1e-4,'yEcho':1e-1,'yInfer':1e-4}
vars = ['yHat','yEcho','yInfer']
print("Tolerances are",tol)
print("---------------------------------------------------------------")

print("Test 1: ", end='')
for i in vars:
    normErr = np.linalg.norm(ref1[i]-cur1[i])/M/D
    # print('Normalized error of',i,'is',normErr)
    if normErr < tol[i]:
        print('\tPASSED', end='')
    else:
        print('\tFAILED', end='')
print("\n---------------------------------------------------------------")

print("Test 2: ", end='')
for i in vars:
    normErr = np.linalg.norm(ref2[i]-cur2[i])/M/D
    # print('Normalized error of',i,'is',normErr)
    if normErr < tol[i]:
        print('\tPASSED', end='')
    else:
        print('\tFAILED', end='')
print("\n---------------------------------------------------------------")

print("Test 3: ", end='')
for i in vars:
    normErr = np.linalg.norm(ref3[i]-cur3[i])/M/D
    # print('Normalized error of',i,'is',normErr)
    if normErr < tol[i]:
        print('\tPASSED', end='')
    else:
        print('\tFAILED', end='')
print("\n---------------------------------------------------------------")

print("Test 4: ", end='')
for i in vars:
    normErr = np.linalg.norm(ref4[i]-cur4[i])/M/D
    # print('Normalized error of',i,'is',normErr)
    if normErr < tol[i]:
        print('\tPASSED', end='')
    else:
        print('\tFAILED', end='')
print("\n---------------------------------------------------------------")