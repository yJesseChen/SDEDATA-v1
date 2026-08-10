import numpy as np
import sys
import os
import scipy.io as sio
import pdb

def std_normal(N_data, t_steps):
    # x0 and t_steps should be 1d array
    diff = t_steps[1:]-t_steps[:-1]
    grow = np.zeros([N_data,t_steps.shape[0]])
    for i in range(t_steps.shape[0]-1):
        grow[:,i+1] = np.random.normal(0.0, np.sqrt(diff[i]), N_data)
    return grow

def EM_auto_1d(drift,diffusion,initial,t_steps):
    data = np.zeros([initial.shape[0],t_steps.shape[0]])
    data[:,0] = initial
    noise = std_normal(initial.shape[0], t_steps-1)
    diff = t_steps[1:]-t_steps[:-1]
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i]
        data[:,i+1] = Xt+drift(Xt)*diff[i]+diffusion(Xt)*noise[:,i+1]
    return data

def geneq(sigma):
    def drift(x):
        return x-x**3
    def diff(x):
        return sigma
    return drift,diff

def Gendata(T,Nt,N_data,IC_,mean=False,steady=False):
    #
    #
    # The Geometric Brownian Motion:
    # dX_t = X_t-X_t^3 dt + sigma dB_t
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    # 
    sigma = 0.5
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        xIC = np.random.uniform(-2.5,2.5,N_data)
    elif IC_=='value':
        xIC = 1.5*np.ones(N_data)
    # function
    drift,diffu = geneq(sigma)
    # data
    data = np.zeros((1,Nt+1,N_data))
    data[0,:,:] = (EM_auto_1d(drift,diffu,xIC,t)).T
    if steady:
        Nt = 1
        data = data[:,[0,-1],:]
        data = np.tile(data,(20,1,1))
        for i in range(20):
            data[i,0,:] = (data[i,0,:])[np.random.permutation(N_data)]
            data[i,1,:] = (data[i,1,:])[np.random.permutation(N_data)]
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
    data_train = Gendata(T=0.01, Nt=1, N_data=10000, IC_='uniform',steady=False)
    data_test  = Gendata(T=10.0, Nt=1000, N_data=5000,  IC_='value',mean=False,steady=False)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})
