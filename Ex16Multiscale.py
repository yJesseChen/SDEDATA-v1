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

# def EM(x0,f,diff,tseq):
#     numT = tseq.shape[0]
#     dtseq = tseq[1:]-tseq[:-1]
#     re = np.zeros([x0.shape[0],numT])
#     re[:,0] = x0
#     noise = std_normal(x0.shape[0], tseq)
#     for i in range(numT-1):
#         re[:,i+1] = re[:,i]+dtseq[i]*f(re[:,i],tseq[i])+diff(re[:,i],tseq[i])*noise[:,i+1]
#     return re

def EM_md(drift,diffusion,dim,initial,t_steps):
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = t_steps[1:]-t_steps[:-1]
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i*dim:(i+1)*dim]
        data[:,(i+1)*dim:(i+2)*dim] = Xt+drift(Xt,t_steps[i])*diff[i]+((noise[:,(i+1)*dim:(i+2)*dim][:,None,:])@(diffusion(Xt,t_steps[i])))[:,0,:]
    datagen = np.zeros((dim,t_steps.shape[0],initial.shape[0]))
    for i in range(dim):
        datagen[i,:,:] = (data[:,i::dim]).T
    return datagen

def EMgamma_md(drift,diffusion,dim,initial,para,t_steps):
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = t_steps[1:]-t_steps[:-1]
    dt = diff[0]
    b_ = gamma2(dt,para[0].T,para[1].T,para[2].T)
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i*dim:(i+1)*dim]
        data[:,(i+1)*dim:(i+2)*dim] = Xt+drift(Xt,b_[:,i])*diff[i]+((noise[:,(i+1)*dim:(i+2)*dim][:,None,:])@(diffusion(Xt,b_[:,i])))[:,0,:]
    datagen = np.zeros((dim,t_steps.shape[0],initial.shape[0]))
    for i in range(dim):
        datagen[i,:,:] = (data[:,i::dim]).T
    return datagen

# def EMgamma_md(x0,para,f,diff,tseq):
#     numT = tseq.shape[0]
#     dtseq = tseq[1:]-tseq[:-1]
#     dt = tseq[1]-tseq[0]
#     re = np.zeros([x0.shape[0],numT])
#     re[:,0] = x0
#     noise = std_normal(x0.shape[0], tseq)
#     b_ = gamma2(dt,para[0].T,para[1].T,para[2].T)
#     for i in range(numT-1):
#         re[:,i+1] = re[:,i]+dtseq[i]*f(re[:,i],b_[:,i])+diff(re[:,i],b_[:,i])*noise[:,i+1]
#     return re

def fgen(s1,s2):
    def b(t):
        return np.sin(np.pi*t)+np.cos(np.sqrt(2)*np.pi*t)
    def f_exact(x,t):
        return np.array((-x[:,1]**3+b(t),-(x[:,1]-x[:,0]))).T
    def f_appx(x,b_):
        return np.array((-x[:,1]**3+b_,-(x[:,1]-x[:,0]))).T
    def diff_exact(x,t):
        sigma = np.array(((s1,0),(0,s2)))
        return np.repeat(sigma[None,:,:],x.shape[0],axis=0)
    def diff_appx(x,c_):
        sigma = np.array(((s1,0),(0,s2)))
        return np.repeat(sigma[None,:,:],x.shape[0],axis=0)
    return f_exact,f_appx,b,diff_exact,diff_appx

def gamma2(t,G1,G2,G3):
    return G1+G2*t+G3*t**2

def Gamma2(tsq,b):
    # given tseq t0, t1, t2,...
    # solve Gamma0, Gamma1,... of local approximation wpt b
    numT = tsq.shape[0]
    dt = tsq[1]-tsq[0]
    re = np.zeros([3,numT-1])
    V  = np.vander(np.array((0,dt/2,dt)), increasing=True)
    Vi = np.linalg.inv(V)
    for i in range(numT-1):
        re[:,i] = np.dot(Vi,b(np.array((tsq[i],tsq[i]+dt/2,tsq[i]+dt))))
    return re

def Gendata(T,Nt,N_data,ifrandom,ifpara):
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
    s1 = 0.2
    s2 = 0.05
    f_exact,f_appx,b,diff_exact,diff_appx = fgen(s1,s2)
    dim = 2
    # initial condition - can be changed
    if ifrandom:
        # xIC = np.random.uniform(-2,2,N_data)
        xIC = np.array((np.random.uniform(-1.5,2,N_data),np.random.uniform(-1.0,1.6,N_data))).T
    else:
        xIC = np.array((2.0*np.ones(N_data),1.0*np.ones(N_data))).T
    # data = np.zeros([1,Nt+1,N_data])
    if not ifpara:
        dim_para     = 1
        data         = EM_md(f_exact,diff_exact,dim,xIC,t)
        para_data    = np.zeros([1,Nt,N_data])
        para_data[0] = (np.tile(t[:-1],N_data).reshape([N_data,Nt])).T
    else:
        dim_para     = 3
        if ifrandom:
            para_data = np.array((np.random.uniform(-2,2,[Nt,N_data]),np.random.uniform(-8,8,[Nt,N_data]),np.random.uniform(-15,15,[Nt,N_data])))
            data      = EMgamma_md(f_appx,diff_appx,dim,xIC,para_data,t)
        # if ifrandom:
        #     # para_data_i = np.random.uniform(-9,9,[3,N_data])
        #     para_data_i = np.random.uniform(-10,10,[3,N_data])
        #     para_data = uniform_seq(para_data_i,t[:-1],[-9,9]) # note here
        #     data[0]   = (EMgamma(xIC,para_data,f_appx,diff_appx,t)).T
        else:
            data   = EM_md(f_exact,diff_exact,dim,xIC,t)
            para_data = Gamma2(t,b)
    # data
    if N_data==1:
        data      = data.reshape([1,Nt+1])
        para_data = para_data.reshape([dim_para,Nt])
    return data,para_data

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    data_train, para_train = Gendata(T=1.0,    Nt=100,    N_data=10000,  ifrandom=True, ifpara=True)
    data_test , para_test  = Gendata(T=10.0,   Nt=1000,   N_data=10000,  ifrandom=False,ifpara=True)
    sio.savemat(traindatapath,{'data':data_train,'para':para_train})
    sio.savemat(testdatapath ,{'data':data_test ,'para':para_test })
