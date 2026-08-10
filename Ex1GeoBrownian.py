import numpy as np
import sys
import os
import scipy.io as sio
import pdb

def std_brownian(N_data, t_steps):
    # x0 and t_steps should be 1d array
    diff = t_steps[1:]-t_steps[:-1]
    grow = np.zeros([N_data,t_steps.shape[0]])
    for i in range(t_steps.shape[0]-1):
        grow[:,i+1] = np.random.normal(0.0, np.sqrt(diff[i]), N_data)
    grow = np.cumsum(grow, axis=-1)
    return grow

def Gendata(T,Nt,N_data,IC_,mean=False):
    #
    #
    # The Geometric Brownian Motion:
    # dX_t = mu X_t dt + sigma X_t dB_t
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    # 
    # mu = 0.2
    # sigma = 0.5
    mu = 2.0
    sigma = 1.0
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        xIC = np.random.uniform(0,10,N_data)
    elif IC_=='value':
        xIC = 0.5*np.ones(N_data)
    # data
    data = np.zeros((1,Nt+1,N_data))
    brownian = std_brownian(N_data, t)
    data[0,:,:] = (xIC[:,None]*np.exp((mu-sigma**2/2)*t+sigma*brownian)).T
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
    data_train = Gendata(T=0.01, Nt=1, N_data=100000, IC_='uniform')
    data_test  = Gendata(T=1.0, Nt=100, N_data=1000,  IC_='value',mean=False)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})
