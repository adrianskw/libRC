#%% -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:56:30 2022

@author: Adrian Wong
"""
# %reload_ext autoreload
# %autoreload 2

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

M = 30000
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


# ds = 0.3
# RC = diffRC(N,D,ds,bias=True)
# RC.chooseIntegrator('RK2')

RC = mapRC(N,D,bias=True)

RC.makeConnectionMat(rho,loc=-1,scale=2)
RC.makeInputMat(sigma,randMin=0,randMax=1)

# RC.A = sp.sparse.eye(N)

RC.listen(y)
RC.train(y,alpha=0.1)
RC.echo(Mpred)

driveIndex =[0,1,2]
measInterval = 20
RC.infer(yPred[driveIndex],driveIndex)
RC.inferImplicit(yPred[driveIndex],driveIndex,measInterval)

plt.rcParams['figure.figsize'] = [15, 3]
plt.plot(xPred[0,:Mplot],'.')
plt.plot(RC.yInferImplicit[0,:Mplot])
plt.plot(RC.yInfer[0,:Mplot])
plt.show()

ticks = measInterval*(np.arange(int(Mpred/measInterval)))
plt.semilogy(np.abs(yPred[0,:Mplot]-RC.yInferImplicit[0,:Mplot]))
plt.semilogy(ticks,np.abs(yPred[0,:Mplot:measInterval]-RC.yInferImplicit[0,:Mplot:measInterval]),'.')
plt.show()
print(np.linalg.norm(yPred-RC.yInferImplicit)/M)

plt.semilogy(np.linalg.norm(RC.rInfer[:,:Mplot]-RC.rInferImplicit[:,:Mplot],axis=0))
plt.semilogy(ticks,np.linalg.norm(RC.rInfer[:,:Mplot:measInterval]-RC.rInferImplicit[:,:Mplot:measInterval],axis=0),'.')
plt.show()
print(np.linalg.norm(RC.rInfer-RC.rInferImplicit)/M)
#%%
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.figsize'] = [15, 10]
fig,axs = plt.subplots(3,sharex='col')
fig.subplots_adjust(hspace=0)
labelList = ['x','y','z']
for i in range(D):
    axs[i].plot(yPred[i,:Mplot],label='data')
    axs[i].plot(RC.yEcho[i,:Mplot],label='echo')
    axs[i].plot(RC.yInfer[i,:Mplot],label='infer')
    axs[i].plot(RC.yInferImplicit[i,:Mplot],label='implicit')
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
plt.plot(RC.yInferImplicit[0],RC.yInferImplicit[1],label='implicit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Attractors')
plt.legend()
plt.show()

PCmat = RC.inferPC(yPred)
plt.rcParams['figure.figsize'] = [8, 8]
plt.plot(yPred[0],RC.yInferImplicit[0],',',label='implicit')
plt.plot(yPred[0],RC.yInfer[0],',',label='infer')
plt.xlabel('data')
plt.ylabel('infer')
plt.legend()
plt.title(f'Reconstruction PC = {PCmat[0,0]:0.3f}')
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
# %%