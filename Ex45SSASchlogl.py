import numpy as np
import sys
import os
import scipy
import scipy.io as sio
import pdb
import time
import psutil
import gc


def harzard(A,c):
    def f(x):
        re = np.array((x*(x-1)/2,x*(x-1)*(x-2)/6,1,x))
        return c*re
    return f

def GillespieSSA(A,B,c,initial,T):
    # Shape: initial: [1,         N_species]
    #        A,B    : [N_species, N_reaction]
    #        c      : [1,         N_reaction]
    N_species = 1
    S = B-A
    react_time = [0]
    react_numb = [initial]
    harzardf = harzard(A,c)
    x = initial
    t = 0
    while t<T:
        a = harzardf(x)
        a0 = np.sum(a)
        # print(t)
        if abs(a0)<1e-8:
            react_time.append(T+1)
            react_numb.append(x)
            break
        tau = np.random.exponential(1/a0)
        r2 = np.random.uniform(0,1)
        r = findinteger(np.cumsum(a)/a0,r2)
        x = x + S[int(r)]
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
    dim = 1
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

def GillespieSSA_MemFriendly(A,B,c,initial,t_step):
    # Shape: initial: [1,         N_species]
    #        A,B    : [N_species, N_reaction]
    #        c      : [1,         N_reaction]
    N_species = 1
    T = t_step[-1]
    NT = t_step.shape[0]
    re = np.zeros([N_species,NT])
    # record the results every N_record step of t_step
    N_record = 400
    count = 1

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
        x = x + S[int(r)]
        t = t+tau
        react_time.append(t)
        react_numb.append(x)

        # print(t)

        # record
        if t>t_step[min(int(N_record*count),NT)-1]:
            print('re')
            # print(len(react_time))
            react_time = np.array(react_time)
            react_numb = np.vstack(react_numb)

            st = int((count-1)*N_record)
            ed = int(count*N_record)
            for j in range(N_species):
                f = scipy.interpolate.interp1d(react_time,react_numb[:,j],kind='previous')
                re[j,st:ed] = f(t_step[st:ed])
            
            react_time = [t]
            react_numb = [x]
            count += 1
    
    return re

# def GenDatawithGillespieSSA(A,B,c,initial,t_step):
#     dim = 1
#     N_data = initial.shape[0]
#     data_re = np.zeros([dim,t_step.shape[0],N_data])
#     for i in range(N_data):
#         st = time.time()
#         print(i)
#         data_re[:,:,i] = GillespieSSA_MemFriendly(A,B,c,initial[i],t_step)
#         ### for memory
#         # del f
#         # del react_time
#         # del react_numb
#         # gc.collect()
#         ### for memory
#         # print(time.time()-st)
#         # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

#         # os.chdir(sys.path[0])
#         # filename = (sys.argv[0].split('/')[-1].split('.')[0])
#         # testdatapath  = '../'+filename+'_test.mat'
#         # sio.savemat(testdatapath ,{'data':data_re})
#     return data_re

def GillespieSSAOriginal(A,B,c,initial,t_step):
    dim = 1
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
        # print(time.time()-st)
        # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    return data_re

def GillespieSSA_stopptime(A,B,c,initial,creteria,maxT):
    # Shape: initial: [1,         N_species]
    #        A,B    : [N_species, N_reaction]
    #        c      : [1,         N_reaction]
    N_species = 1
    S = B-A
    harzardf = harzard(A,c)
    x = initial
    t = 0
    while t<maxT:
        a = harzardf(x)
        a0 = np.sum(a)
        tau = np.random.exponential(1/a0)
        r2 = np.random.uniform(0,1)
        r = findinteger(np.cumsum(a)/a0,r2)
        x = x + S[int(r)]
        t = t+tau
        if creteria(x):
            break
    re_t = min(maxT,t)
    print(re_t)
    return re_t


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
    dim = 1

    # Set 1 - Glispie
    A = np.array((2,3,0,1))
    B = np.array((3,2,1,0))
    c = np.array((3e-2,1e-4,200,3.5))
    x1r = [0,750]
    x1d = 250
    
    # initial condition - can be changed
    if ifrandom:
        xIC = (np.random.randint(x1r[0],x1r[1],N_data))
    else:
        xIC = (x1d*np.ones(N_data,dtype='int'))
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
    dim = 1

    # Set 1 - Glispie
    A = np.array((2,3,0,1))
    B = np.array((3,2,1,0))
    c = np.array((3e-2,1e-4,200,3.5))
    x1r = [0,750]
    x1d = 250
    # x_in = np.random.randint(750,size=16)
    # x_in = np.random.randint(200,300,size=4)
    # x_in = np.array([289,77,713,46,662,499,464,643,633,117,224,207,227,701,101,609])
    x_in = np.array([245,246,247,248,249,250,251,252])
    
    data_dic = {}
    for j in range(x_in.shape[0]):
        x1d = x_in[j]
        xIC = x1d*np.ones(N_data,dtype='int')
        data = ((GenDatawithGillespieSSA(A,B,c,xIC,t))[:,1,:]).T
        data_dic[str(j)+'_i'] = x_in[j]
        data_dic[str(j)+'_d'] = data
    data_dic['size'] = np.array((np.sqrt(x_in.shape[0]),np.sqrt(x_in.shape[0])))

    # K = 40
    # x_in = np.linspace(50,600,K).astype('int')
    # # x_in = np.array([289,77,713,46,662,499,464,643,633,117,224,207,227,701,101,609])
    # re = np.zeros(x_in.shape[0])
    # re2 = np.zeros(x_in.shape[0])
    # for j in range(x_in.shape[0]):
    #     x1d = x_in[j]
    #     xIC = x1d*np.ones(N_data,dtype='int')
    #     data = ((GenDatawithGillespieSSA(A,B,c,xIC,t))[:,1,:]).T
    #     re[j] = np.std(data.flatten())/np.sqrt(T)
    #     re2[j] = (np.mean(data.flatten())-x1d)/T
    #     # re[j] = np.std(data.flatten())
    #     # re2[j] = (np.mean(data.flatten())-x1d)
    # print(re)
    # print(re2)
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
    dim = 1

    # Set 1 - Glispie
    A = np.array((2,3,0,1))
    B = np.array((3,2,1,0))
    c = np.array((3e-2,1e-4,200,3.5))
    x1r = [563,564]
    x1d = 250
    
    # xIC = x1d*np.ones(N_data,dtype='int')
    xIC = (np.random.randint(x1r[0],x1r[1],N_data))
    data = GillespieSSAOriginal(A,B,c,xIC,t)
    return data

def Gendata_stoppingtime(T,Nt,N_data):
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
    dim = 1

    # Set 1 - Glispie
    A = np.array((2,3,0,1))
    B = np.array((3,2,1,0))
    c = np.array((3e-2,1e-4,200,3.5))
    x1r = [0,750]
    x1d = 250
    x_in = np.array((82,563))

    data_dic = {}
    exitcond = [lambda x: x>=563, lambda x: x<=82]
    for j in range(x_in.shape[0]):
        data = []
        count = 0
        for i in range(N_data):
            print(i)
            data.append(GillespieSSA_stopptime(A,B,c,x_in[j],exitcond[j],200000))
        data_dic[str(j)+'_i'] = x_in[j]
        data_dic[str(j)+'_d'] = np.array(data)

    return data_dic

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    # data_train = Gendata(T=0.01,   Nt=1,      N_data=200000,  ifrandom=True)
    # sio.savemat(traindatapath,{'data':data_train})
    # data_test  = Gendata(T=50.0,   Nt=5000,   N_data=10000,  ifrandom=False)
    # sio.savemat(testdatapath ,{'data':data_test})
    
    # conddatapath  = '../'+filename+'_cond.mat'
    # data_dic = Gendata_condition(T=0.02,   Nt=1,      N_data=10000)
    # sio.savemat(conddatapath,data_dic)

    oridatapath  = '../'+filename+'_ori.mat'
    data_dic = GendataOri(T=50,   Nt=500,      N_data=2)
    sio.savemat(oridatapath,data_dic)

    # stoppingtimedatapath  = '../'+filename+'_stoppingtime.mat'
    # data_dic = Gendata_stoppingtime(T=0.1,   Nt=1,  N_data=1000)
    # sio.savemat(stoppingtimedatapath,data_dic)
