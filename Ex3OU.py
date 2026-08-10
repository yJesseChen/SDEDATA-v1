import numpy as np
import sys
import os
import scipy.io as sio
import pdb

def std_normal(N_data, t_steps, seeds):
    # x0 and t_steps should be 1d array
    np.random.seed(seeds)
    diff = t_steps[1:]-t_steps[:-1]
    grow = np.zeros([N_data,t_steps.shape[0]])
    noise = np.random.normal(0.0, np.sqrt(diff[0]), [t_steps.shape[0]-1, N_data])
    for i in range(t_steps.shape[0]-1):
        grow[:,i+1] = noise[i]
    return grow

def Gendata(T,Nt,N_data,IC_,seeds,mean=False,steady=False):
    #
    #
    # The Ornstein-Uhlenbeck process:
    # dX_t = th(mu-X_t) dt + sig dB_t
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    # 
    th = 1.0
    mu = 1.2
    sig = 0.3
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        np.random.seed(2)
        xIC = np.random.uniform(0,2.5,N_data)
    elif IC_=='value':
        xIC = 1.5*np.ones(N_data)
    # data
    data = np.zeros((1,Nt+1,N_data))
    brownian = std_normal(N_data, t, seeds)
    Ext = np.exp(-th*t)
    data[0,:,:] = (xIC[:,None]*Ext+mu*(1-Ext)+sig*Ext*np.cumsum(np.exp(th*t)*brownian, axis=-1)).T
    if steady:
        Nt = 1
        data = data[:,[0,-1],:]
        data[0][1] -= np.mean(data[0][1])
        # data = np.tile(data,(10,1,1))
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
    data_train = Gendata(T=0.01, Nt=1, N_data=1000000, IC_='uniform', seeds=1,steady=False)
    data_test  = Gendata(T=5.0, Nt=500, N_data=100000,  IC_='value', seeds=1,mean=False,steady=False)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})
