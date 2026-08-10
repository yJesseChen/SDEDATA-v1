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

def uniform_seq(x0,tseq,range_):
    numT = tseq.shape[0]
    dtseq = tseq[1:]-tseq[:-1]
    re = np.zeros([x0.shape[0],numT,x0.shape[1]])
    re[:,0,:] = x0
    noise = np.random.uniform(range_[0],range_[1],[x0.shape[0], tseq.shape[0], x0.shape[1]])
    for i in range(numT-1):
        re[:,i+1,:] = re[:,i,:]+dtseq[0]*noise[:,i+1,:]
    return re

def EM(x0,f,diff,tseq):
    numT = tseq.shape[0]
    dtseq = tseq[1:]-tseq[:-1]
    re = np.zeros([x0.shape[0],numT])
    re[:,0] = x0
    noise = std_normal(x0.shape[0], tseq)
    for i in range(numT-1):
        re[:,i+1] = re[:,i]+dtseq[i]*f(re[:,i],tseq[i])+diff(re[:,i],tseq[i])*noise[:,i+1]
    return re

def EMgamma(x0,para,f,diff,tseq):
    numT = tseq.shape[0]
    dtseq = tseq[1:]-tseq[:-1]
    dt = tseq[1]-tseq[0]
    re = np.zeros([x0.shape[0],numT])
    re[:,0] = x0
    noise = std_normal(x0.shape[0], tseq)
    b_ = gamma2(dt,para[0].T)
    for i in range(numT-1):
        re[:,i+1] = re[:,i]+dtseq[i]*f(re[:,i],b_[:,i])+diff(re[:,i],b_[:,i])*noise[:,i+1]
    return re

def fgen(omega,sigma,V):
    def b(t):
        return V*np.cos(omega*t)
    # def b(t):
    #     return 0
    def f_exact(x,t):
        return (x-x**3)+b(t)
    def f_appx(x,b_):
        return (x-x**3)+b_
    def diff_exact(x,t):
        return sigma
    def diff_appx(x,c_):
        return sigma
    return f_exact,f_appx,b,diff_exact,diff_appx

def gamma2(t,G1):
    return G1

# def Gamma2(tsq,b):
#     # given tseq t0, t1, t2,...
#     # solve Gamma0, Gamma1,... of local approximation wpt b
#     numT = tsq.shape[0]
#     dt = tsq[1]-tsq[0]
#     re = np.zeros([3,numT-1])
#     V  = np.vander(np.array((0,dt/2,dt)), increasing=True)
#     Vi = np.linalg.inv(V)
#     for i in range(numT-1):
#         re[:,i] = np.dot(Vi,b(np.array((tsq[i],tsq[i]+dt/2,tsq[i]+dt))))
#     return re

def Gamma2(tsq,b):
    # given tseq t0, t1, t2,...
    # solve Gamma0, Gamma1,... of local approximation wpt b
    numT = tsq.shape[0]
    dt = tsq[1]-tsq[0]
    re = np.zeros([1,numT-1])
    for i in range(numT-1):
        re[0,i] = b(tsq[i])
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
    sigma = 0.25
    omega = 0.001
    V     = 0.12
    f_exact,f_appx,b,diff_exact,diff_appx = fgen(omega,sigma,V)
    # initial condition - can be changed
    if ifrandom:
        xIC = np.random.uniform(-1.6,1.6,N_data)
    else:
        xIC = 1.0*np.ones(N_data)
    data = np.zeros([1,Nt+1,N_data])
    if not ifpara:
        dim_para     = 1
        data[0]      = (EM(xIC,f_exact,diff_exact,t)).T
        para_data    = np.zeros([1,Nt,N_data])
        para_data[0] = (np.tile(t[:-1],N_data).reshape([N_data,Nt])).T
    else:
        dim_para     = 1
        if ifrandom:
            para_data = np.random.uniform(-0.13,0.13,[1,Nt,N_data])
            data[0]   = (EMgamma(xIC,para_data,f_appx,diff_appx,t)).T
        # if ifrandom:
        #     # para_data_i = np.random.uniform(-9,9,[3,N_data])
        #     para_data_i = np.random.uniform(-10,10,[3,N_data])
        #     para_data = uniform_seq(para_data_i,t[:-1],[-9,9]) # note here
        #     data[0]   = (EMgamma(xIC,para_data,f_appx,diff_appx,t)).T
        else:
            data[0]   = (EM(xIC,f_exact,diff_exact,t)).T
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
    # data_train, para_train = Gendata(T=0.1,      Nt=1,       N_data=200000, ifrandom=True, ifpara=True)
    data_test , para_test  = Gendata(T=40000.0,  Nt=400000,  N_data=500,   ifrandom=False,ifpara=True)
    # sio.savemat(traindatapath,{'data':data_train,'para':para_train})
    sio.savemat(testdatapath ,{'data':data_test ,'para':para_test })
