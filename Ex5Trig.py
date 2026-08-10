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

def geneq(k,sigma):
    def drift(x):
        return np.sin(2*k*np.pi*x)
    def diff(x):
        return sigma*np.cos(2*k*np.pi*x)
    return drift,diff

def Gendata(T,Nt,N_data,IC_,mean=False):
    #
    #
    # The Ito SDE with trig drift and diffusion:
    # dX_t = sin(2k\pi X_t) dt + sigma cos(2k\pi X_t) dB_t
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    # 
    k = 1
    sigma = 0.5
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        xIC = np.random.uniform(0.35,0.7,N_data)
    elif IC_=='value':
        xIC = 0.6*np.ones(N_data)
    # function
    drift,diffu = geneq(k,sigma)
    # data
    data = np.zeros((1,Nt+1,N_data))
    data[0,:,:] = (EM_auto_1d(drift,diffu,xIC,t)).T
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
    data_test  = Gendata(T=10.0, Nt=1000, N_data=100000,  IC_='value',mean=False)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})
