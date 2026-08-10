import numpy as np
import sys
import os
import scipy.io as sio
import pdb

def std_exp(N_data, t_steps):
    # x0 and t_steps should be 1d array
    diff = t_steps[1:]-t_steps[:-1]
    grow = np.zeros([N_data,t_steps.shape[0]])
    for i in range(t_steps.shape[0]-1):
        grow[:,i+1] = np.random.exponential(1.0, N_data)
    return grow

def EM_auto_1d(condprob,initial,t_steps):
    data = np.zeros([initial.shape[0],t_steps.shape[0]])
    data[:,0] = initial
    noise = std_exp(initial.shape[0], t_steps-1)
    diff = t_steps[1:]-t_steps[:-1]
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i]
        data[:,i+1] = condprob(Xt,diff[i],noise[:,i+1])
    return data

def geneq(th,sig):
    def condprob(x,dt,dw):
        return x+th*x*dt+sig*dw*np.sqrt(dt)
    return condprob

def Gendata(T,Nt,N_data,IC_,mean=False):
    #
    #
    # The Geometric Brownian Motion:
    # dX_t = mu X_t dt + sigma * Exp(1) * sqrt(dt)
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps 
    # N_data : number of data trajectories
    # 
    th =  -2.0
    sig = 0.1
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        xIC = np.random.uniform(0.2,0.9,N_data)
    elif IC_=='value':
        xIC = 0.4*np.ones(N_data)
    # function
    condprob = geneq(th,sig)
    # data
    data = np.zeros((1,Nt+1,N_data))
    data[0,:,:] = (EM_auto_1d(condprob,xIC,t)).T
    if mean:
        data = (np.mean(data,axis=2)).reshape([1,Nt+1])
    if N_data==1:
        data = data.reshape([1,Nt+1])
    return data

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    data_train = Gendata(T=1.0, Nt=100, N_data=10000, IC_='uniform')
    data_test  = Gendata(T=5.0, Nt=500, N_data=100000,  IC_='value',mean=False)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})
