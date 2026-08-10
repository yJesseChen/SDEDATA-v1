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

def uniform_seq(x0,tseq,range_):
    numT = tseq.shape[0]
    dtseq = tseq[1:]-tseq[:-1]
    re = np.zeros([x0.shape[0],numT,x0.shape[1]])
    re[:,0,:] = x0
    noise = np.random.uniform(range_[0],range_[1],[x0.shape[0], tseq.shape[0], x0.shape[1]])
    for i in range(numT-1):
        re[:,i+1,:] = re[:,i,:]+dtseq[0]*noise[:,i+1,:]
    return re

def EM_md(drift,diffusion,dim,initial,t_steps):
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = t_steps[1:]-t_steps[:-1]
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i*dim:(i+1)*dim]
        data[:,(i+1)*dim:(i+2)*dim] = Xt+drift(Xt)*diff[i]+((noise[:,(i+1)*dim:(i+2)*dim][:,None,:])@(diffusion(Xt)))[:,0,:]
    datagen = np.zeros((dim,t_steps.shape[0],initial.shape[0]))
    for i in range(dim):
        datagen[i,:,:] = (data[:,i::dim]).T
    return datagen

def EM_md_refine(drift,diffusion,dim,initial,t_steps,Multipli_):
    # only work for constant time step
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = (t_steps[1:]-t_steps[:-1])[0]/Multipli_
    Xt = initial
    for i in range(t_steps.shape[0]-1):
        for _ in range(Multipli_):
            noise = np.random.normal(0.0, np.sqrt(diff), [initial.shape[0],dim])
            Xnew = Xt+drift(Xt)*diff+((noise[:,None,:])@(diffusion(Xt)))[:,0,:]
            Xt = Xnew
        data[:,(i+1)*dim:(i+2)*dim] = Xt
    datagen = np.zeros((dim,t_steps.shape[0],initial.shape[0]))
    for i in range(dim):
        datagen[i,:,:] = (data[:,i::dim]).T
    return datagen

def fgen(epsilon,sigma_1,sigma_2,sigma_3):
    def f_exact(x):
        return np.array((x[:,1],-x[:,1]+x[:,2]**2-x[:,0],1/epsilon*(1/4*x[:,0]-x[:,2]))).T
    def diff_exact(x):
        sigma = np.array(((sigma_1,0,0),(0,sigma_2,0),(0,0,sigma_3*np.sqrt(2)/np.sqrt(epsilon))))
        return np.repeat(sigma[None,:,:],x.shape[0],axis=0)
    return f_exact,diff_exact

def Gendata(T,Nt,N_data,ifrandom,reduced=False):
    #
    #
    # The ode system:
    # x'   = -a(t)x+b(t)
    # x[0] = x0
    #
    #
    #
    # Parameters:
    # Nt    : number of discretized t steps
    # N_data : number of data trajectories
    # 
    t = np.linspace(0,T,Nt+1)
    epsilon = 0.001
    sigma_1 = 0.3
    sigma_2 = 0.3
    sigma_3 = 0.1
    f_exact,diff_exact = fgen(epsilon,sigma_1,sigma_2,sigma_3)
    dim = 3
    # initial condition - can be changed
    if ifrandom:
        # xIC = np.random.uniform(-2,2,N_data)
        xIC = np.array((np.random.uniform(-1.5,2.5,N_data),np.random.uniform(-2.0,1.5,N_data),np.random.uniform(-0.6,1.0,N_data))).T
    else:
        # xIC = np.array((1.5*np.ones(N_data),2.0*np.ones(N_data))).T
        xIC = np.array((1.5*np.ones(N_data),1.0*np.ones(N_data),np.random.normal(1.5/4,sigma_3,N_data))).T
    # data         = EM_md(f_exact,diff_exact,dim,xIC,t)
    data         = EM_md_refine(f_exact,diff_exact,dim,xIC,t,100)
    # data
    if reduced:
        data = data[:2,:,:]
    if N_data==1:
        data      = data.reshape([1,Nt+1])
    return data

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    data_train = Gendata(T=1.0, Nt=100, N_data=10000,  ifrandom=True, reduced=True)
    data_test  = Gendata(T=8.0, Nt=800, N_data=10000,  ifrandom=False, reduced=True)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})

