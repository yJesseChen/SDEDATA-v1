import numpy as np
import sys
import os
import scipy
import scipy.io as sio
import pdb
import time
import psutil
import gc

# See http://masa.o.oo7.jp/gillespie/


def harzard(A,c):
    def f(x):
        re = np.array((x[0]*x[1],x[1]*x[2],x[0]*x[2]))
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
        if abs(a0)<1e-8:
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
        st = time.time()
        print(i)
        react_time,react_numb = GillespieSSA(A,B,c,initial[i],t_step[-1])
        for j in range(dim):
            f = scipy.interpolate.interp1d(react_time,react_numb[:,j],kind='previous')
            data_re[j,:,i] = f(t_step)
        ### for memory
        # del f
        # del react_time
        # del react_numb
        # gc.collect()
        ### for memory
        print(time.time()-st)
        # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        # os.chdir(sys.path[0])
        # filename = (sys.argv[0].split('/')[-1].split('.')[0])
        # testdatapath  = '../'+filename+'_test.mat'
        # sio.savemat(testdatapath ,{'data':data_re})
    return data_re

def GillespieSSAOriginal(A,B,c,initial,t_step):
    dim = initial.shape[1]
    N_data = initial.shape[0]
    data_re = {}
    for i in range(N_data):
        st = time.time()
        print(i)
        react_time,react_numb = GillespieSSA(A,B,c,initial[i],t_step[-1])
        data_re['t_'+str(i)] = react_time
        data_re['d_'+str(i)] = react_numb
        ### for memory
        # del f
        # del react_time
        # del react_numb
        # gc.collect()
        ### for memory
        print(time.time()-st)
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

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
    dim = 3

    # Set 1 - Self-setting
    A = np.array(((1,0,1),(1,1,0),(0,1,1)))
    B = np.array(((0,0,2),(2,0,0),(0,2,0)))
    c = np.array((0.002,0.002,0.002))
    x1r,x2r,x3r = [1000,4000],[1000,4000],[1000,4000]
    x1d,x2d,x3d = 3000,2000,2000
    
    # initial condition - can be changed
    if ifrandom:
        xIC = np.array((np.random.randint(x1r[0],x1r[1],N_data),np.random.randint(x2r[0],x2r[1],N_data),np.random.randint(x3r[0],x3r[1],N_data))).T
    else:
        xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data),x3d*np.ones(N_data))).T
    data = GenDatawithGillespieSSA(A,B,c,xIC,t)
    # data
    if N_data==1:
        data      = data.reshape([1,Nt+1])
    return data

def Gendata_condition(T,Nt,N_data):
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
    dim = 3

    # Set 1 - Self-setting
    A = np.array(((1,0,1),(1,1,0),(0,1,1)))
    B = np.array(((0,0,2),(2,0,0),(0,2,0)))
    c = np.array((0.002,0.002,0.002))
    x_in = np.array(((3000,2000,2000),(2500,2500,1800),(2000,3000,2000),(1500,2800,2800),(2000,2000,3200),(2800,2800,1500),(2500,1500,3000),(3000,1800,2300),(0,0,0)))
    
    data_dic = {}
    for j in range(x_in.shape[0]):
        x1d,x2d,x3d = x_in[j]
        xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data),x3d*np.ones(N_data))).T
        data = ((GenDatawithGillespieSSA(A,B,c,xIC,t))[:,1,:]).T
        data_dic[str(j)+'_i'] = x_in[j]
        data_dic[str(j)+'_d'] = data
    data_dic['size'] = np.array((np.sqrt(x_in.shape[0]),np.sqrt(x_in.shape[0])))
    return data_dic

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
    dim = 3

    # Set 1 - Self-setting
    A = np.array(((1,0,1),(1,1,0),(0,1,1)))
    B = np.array(((0,0,2),(2,0,0),(0,2,0)))
    c = np.array((0.002,0.002,0.002))
    x1r,x2r,x3r = [1000,4000],[1000,4000],[1000,4000]
    x1d,x2d,x3d = 3000,2000,2000
    
    # initial condition - can be changed
    xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data),x3d*np.ones(N_data))).T
    data = GillespieSSAOriginal(A,B,c,xIC,t)
    return data

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    # traindatapath = '../'+filename+'_train.mat'
    # testdatapath  = '../'+filename+'_test.mat'
    # data_train = Gendata(T=0.01,   Nt=1,      N_data=200000,  ifrandom=True)
    # sio.savemat(traindatapath,{'data':data_train})
    # data_test  = Gendata(T=10.0,   Nt=1000,   N_data=200,  ifrandom=False)
    # sio.savemat(testdatapath ,{'data':data_test})
    
    # conddatapath  = '../'+filename+'_cond.mat'
    # data_dic = Gendata_condition(T=0.01,   Nt=1,      N_data=500)
    # sio.savemat(conddatapath,data_dic)

    oridatapath  = '../'+filename+'_ori.mat'
    data_dic = GendataOri(T=10.0,   Nt=1000,      N_data=10)
    sio.savemat(oridatapath,data_dic)
