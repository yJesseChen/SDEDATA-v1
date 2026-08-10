import numpy as np
import sys
import os
import scipy
import scipy.io as sio
import pdb
import time


def harzard(A,c):
    N_species  = A.shape[0]
    N_reaction = A.shape[1]
    def f(x):
        re = np.ones(N_reaction,dtype='int')
        for i in range(N_species):
            re = re*scipy.special.comb(x[i],A[i,:])
        return c*re
    return f

def GillespieSSA(A,B,c,initial,T):
    # Shape: initial: [1,         N_species]
    #        A,B    : [N_species, N_reaction]
    #        c      : [1,         N_reaction]
    N_species = initial.shape[0]
    S = B-A
    react_time = [0]
    react_numb = [initial]
    harzardf = harzard(A,c)
    x = initial
    t = 0
    while t<T:
        a = harzardf(x)
        a0 = np.sum(a)
        if a0<1:
            react_time.append(T+1)
            react_numb.append(x)
            break
        tau = np.random.exponential(1/a0)
        r2 = np.random.uniform(0,1)
        r = findinteger(np.cumsum(a)/a0,r2)
        x = x + S[:,int(r)]
        t = t+tau
        react_time.append(t)
        react_numb.append(x)
    react_time = np.array(react_time)
    react_numb = np.vstack(react_numb)
    return react_time,react_numb

def findinteger(a,k):
    # a is an increasing sequence
    # find i such that a[i]<k<a[i+1]
    re = np.where((a[:-1]<k)*(a[1:]>=k))[0]
    if len(re)==1:
        return re[0]+1
    else:
        return 0

def GenDatawithGillespieSSA(A,B,c,initial,t_step):
    dim = initial.shape[1]
    N_data = initial.shape[0]
    data_re = np.zeros([dim,t_step.shape[0],N_data])
    for i in range(N_data):
        print(i)
        react_time,react_numb = GillespieSSA(A,B,c,initial[i],t_step[-1])
        for j in range(dim):
            f = scipy.interpolate.interp1d(react_time,react_numb[:,j],kind='previous')
            data_re[j,:,i] = f(t_step)
    return data_re

# def GenDatawithGillespieSSA(A,B,c,initial,t_step):
#     dim = initial.shape[1]
#     N_data = initial.shape[0]
#     data_re = np.zeros([dim,t_step.shape[0],N_data])
#     for i in range(N_data):
#         print(i)
#         st = time.time()
#         react_time,react_numb = GillespieSSA(A,B,c,initial[i],t_step[-1])
#         print("Time: %.8f"%(time.time()-st))
#         print("Number: %d"%(len(react_time)))
#         for j in range(dim):
#             f = scipy.interpolate.interp1d(react_time,react_numb[:,j],kind='previous')
#             data_re[j,:,i] = f(t_step)
#     return data_re

def GillespieSSAOriginal(A,B,c,initial,t_step):
    dim = initial.shape[1]
    N_data = initial.shape[0]
    data_re = {}
    for i in range(N_data):
        react_time,react_numb = GillespieSSA(A,B,c,initial[i],t_step[-1])
        data_re['t_'+str(i)] = react_time
        data_re['d_'+str(i)] = react_numb

    return data_re


def Gendata(T,Nt,N_data,ifrandom):
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
    A = np.array(((1,0),(0,1),(0,0)))
    B = np.array(((0,0),(1,0),(0,1)))
    c = np.array((1.0,1.0))
    dim = 3
    # initial condition - can be changed
    if ifrandom:
        xIC = np.array((np.random.randint(0,100,N_data),np.random.randint(0,60,N_data),np.random.randint(50,180,N_data))).T
    else:
        xIC = np.array((83*np.ones(N_data,dtype='int'),26*np.ones(N_data),69*np.ones(N_data))).T
    data = GenDatawithGillespieSSA(A,B,c,xIC,t)
    # data
    if N_data==1:
        data      = data.reshape([1,Nt+1])
    return data

def GendataOri(T,Nt,N_data):
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
    A = np.array(((1,0),(0,1),(0,0)))
    B = np.array(((0,0),(1,0),(0,1)))
    c = np.array((1.0,1.0))
    
    # initial condition - can be changed
    xIC = np.array((83*np.ones(N_data,dtype='int'),26*np.ones(N_data),69*np.ones(N_data))).T
    data = GillespieSSAOriginal(A,B,c,xIC,t)
    return data

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    # traindatapath = '../'+filename+'_train.mat'
    # testdatapath  = '../'+filename+'_test.mat'
    # # data_train = Gendata(T=0.1,   Nt=1,      N_data=100000,  ifrandom=True)
    # data_test  = Gendata(T=10.0,   Nt=100,   N_data=100,  ifrandom=False)
    # sio.savemat(traindatapath,{'data':data_train})
    # sio.savemat(testdatapath ,{'data':data_test})

    oridatapath  = '../'+filename+'_ori.mat'
    data_dic = GendataOri(T=10.0,   Nt=100,      N_data=10)
    sio.savemat(oridatapath,data_dic)
