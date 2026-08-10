import numpy as np
import sys
import os
import scipy.io as sio
import pdb

def std_normal(N_data, t_steps, dim):
    # x0 and t_steps should be 1d array
    diff = t_steps[1:]-t_steps[:-1]
    grow = np.zeros([N_data,t_steps.shape[0]*dim])
    for i in range(t_steps.shape[0]-1):
        grow[:,(i+1)*dim:(i+2)*dim] = np.random.normal(0.0, np.sqrt(diff[i]), [N_data,dim])
    return grow

def EM_auto_md(drift,diffusion,dim,initial,t_steps):
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = t_steps[1:]-t_steps[:-1]
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i*dim:(i+1)*dim]
        data[:,(i+1)*dim:(i+2)*dim] = Xt+drift(Xt)*diff[i]+((noise[:,(i+1)*dim:(i+2)*dim][:,None,:])@(diffusion(Xt)))[:,0,:]
    return data

def geneq(dim):
    if dim==2:
        mu = np.array(((-1,-0.5),(-1,-1)))
        sigma = np.array(((1,0),(0,0.5)))
    else:
        mu = -np.diag(1.0*np.ones(dim))
        sigma = np.diag(0.3*np.ones(dim))
    
    def drift(x):
        return x@(mu.T)
    def diff(x):
        return np.repeat(sigma[None,:,:],x.shape[0],axis=0)
    return drift,diff

def Gendata2D(T,Nt,N_data,IC_,mean=False):
    #
    #
    # The Multi-dimension OU:
    # dX_t = mu X_t dt + sigma exp(-X_t^2) dB_t
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    # 
    dim = 5
    t = np.linspace(0,T,Nt+1)
    # initial condition - can be changed
    if IC_=='uniform':
        # xIC = np.array((np.random.uniform(-4,4,N_data),np.random.uniform(-3,3,N_data))).T
        xIC = np.random.uniform(-1,1,[N_data,dim])
    elif IC_=='value':
        xIC = 0.1*np.ones([N_data,dim])
    # function
    drift,diffu = geneq(dim)
    # data
    data = np.zeros((dim,Nt+1,N_data))
    datag = EM_auto_md(drift,diffu,dim,xIC,t)
    for i in range(dim):
        data[i,:,:] = (datag[:,i::dim]).T
    # if steady:
    #     Nt = 1
    #     data = data[:,[0,-1],:]
    #     # data[0][1] -= np.mean(data[0][1])
    #     # data = np.tile(data,(10,1,1))
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
    data_train = Gendata2D(T=0.01, Nt=1, N_data=200000, IC_='uniform')
    data_test  = Gendata2D(T=2.0, Nt=200, N_data=5000,  IC_='value',mean=False)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})
