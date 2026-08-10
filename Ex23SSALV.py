import numpy as np
import sys
import os
import scipy
import scipy.io as sio
import pdb
import time
# import psutil
import gc


def harzard(A,c):
    def f(x):
        re = np.array((x[0],x[0]*x[1],x[1]))
        return c*re
    return f

def GillespieSSA(A,B,c,initial,T):
    # Shape: initial: [1,         N_species]
    #        A,B    : [N_species, N_reaction]
    #        c      : [1,         N_reaction]
    t = 0
    while(t<T):
        N_species = initial.shape[0]
        S = B-A
        react_time = [0]
        react_numb = [initial]
        harzardf = harzard(A,c)
        x = initial
        t = 0
        while t<T:
            a = harzardf(x)
            # print(a)
            a0 = np.sum(a)
            # if abs(a0)<1e-8:
            #     # print('-1')
            #     react_time.append(T+1)
            #     react_numb.append(x)
            #     t = T+1
            #     break
            if abs(x[1])<1e-8:
                print('1')
                # react_time.append(T+1)
                # react_numb.append(x)
                # t = T+1
                break
            if abs(x[0])<1e-8:
                print('2')
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

def GillespieSSAshorttime(A,B,c,initial,T):
    # Shape: initial: [1,         N_species]
    #        A,B    : [N_species, N_reaction]
    #        c      : [1,         N_reaction]
    t = 0
    while(t<T):
        N_species = initial.shape[0]
        S = B-A
        react_time = [0]
        react_numb = [initial]
        harzardf = harzard(A,c)
        x = initial
        t = 0
        while t<T:
            a = harzardf(x)
            # print(a)
            a0 = np.sum(a)
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

def GillespieSSA_stopptime(A,B,c,initial,creteria,maxT):
    # Shape: initial: [1,         N_species]
    #        A,B    : [N_species, N_reaction]
    #        c      : [1,         N_reaction]
    N_species = initial.shape[0]
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
        x = x + S[:,int(r)]
        t = t+tau
        if creteria(x):
            break
    re_t = min(maxT,t)
    return re_t

def findinteger(a,k):
    # a is an increasing sequence
    # find i such that a[i]<k<a[i+1]
    re = np.where((a[:-1]<k)*(a[1:]>=k))[0]
    if len(re)==1:
        return re[0]+1
    else:
        return 0

def GenDatawithGillespieSSA(A,B,c,initial,t_step,longtime=False):
    SSASolver = GillespieSSA if longtime else GillespieSSAshorttime
    dim = initial.shape[1]
    N_data = initial.shape[0]
    data_re = np.zeros([dim,t_step.shape[0],N_data])
    for i in range(N_data):
        st = time.time()
        print(i)
        react_time,react_numb = SSASolver(A,B,c,initial[i],t_step[-1])
        for j in range(dim):
            f = scipy.interpolate.interp1d(react_time,react_numb[:,j],kind='previous')
            data_re[j,:,i] = f(t_step)
        ### for memory
        # del f
        # del react_time
        # del react_numb
        # gc.collect()
        ### for memory
        # print(time.time()-st)
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
        # print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    return data_re


def Gendata(T,Nt,N_data,ifrandom,longtime=False):
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
    dim = 2

    # # Set 1 - Stochastic modelling for systems biology, Wilkinson, Darren James, Fig 6.7
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((1.0,0.005,0.6))
    # x1r,x2r = [0,650],[0,700]
    # x1d,x2d = 100,100

    # # # Set 1 - Stochastic modelling for systems biology, Wilkinson, Darren James, Fig 6.7 - long time T=100
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((1.0,0.005,0.6))
    # x1r,x2r = [0,850],[0,1050]
    # x1d,x2d = 100,100
    # x1LL = [[0,850], [0,300]]
    # x2LL = [[0,1050],[0,300]]
    # prop = [0.5,0.5]

    # Set 1 - low population
    A = np.array(((1,1,0),(0,1,1)))
    B = np.array(((2,0,0),(0,2,0)))
    c = np.array((0.2,0.02,0.2))
    x1r,x2r = [0,90],[0,90]
    x1d,x2d = 10,10
    x1LL = [[0,90],[0,20]]
    x2LL = [[0,90],[0,20]]
    prop = [0.7,0.3]

    # # Set 2 - Glispie
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((10,0.01,10))
    # x1r,x2r = [0,5000],[0,5000]
    # x1d,x2d = 1000,1000
    
    # initial condition - can be changed
    if ifrandom:
        # xIC = np.array((np.random.randint(x1r[0],x1r[1],N_data),np.random.randint(x2r[0],x2r[1],N_data))).T
        xIC = random_blockdata_2d(x1LL,x2LL,prop,N_data)
    else:
        xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data))).T
    data = GenDatawithGillespieSSA(A,B,c,xIC,t,longtime)
    # data
    if N_data==1:
        data      = data.reshape([1,Nt+1])
    return data

def random_blockdata_2d(x1L,x2L,prop,Ndata):
    N = len(x1L)
    dim = len(x1L[0])
    data = np.zeros([Ndata,dim])
    count = 0
    for i in range(N):
        if i==N-1:
            N_d = Ndata-count
        else:
            N_d = int(Ndata*prop[i])
        data[count:count+N_d] = np.array((np.random.randint(x1L[i][0],x1L[i][1],N_d),np.random.randint(x2L[i][0],x2L[i][1],N_d))).T
        count += N_d
    id_ = np.arange(Ndata)
    np.random.shuffle(id_)
    return data[id_]

def Gendata_condition(T,Nt,N_data,longtime=False):
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
    dim = 2

    # Set 1 - Stochastic modelling for systems biology, Wilkinson, Darren James, Fig 6.7
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((1.0,0.005,0.6))
    # x_in = np.array(((100,100),(200,150),(250,250),(150,300),(100,400),(50,400),(20,200),(10,100),(0,0)))
    # x_in = np.array(((100,100),(200,150),(250,250),(150,300),(100,400),(50,400),(20,200),(10,100),(0,80),
    #                  (0,1),(0,2),(0,3),(0,20),(0,100),(0,200),(0,300),(0,600),(0,800),
    #                  (1,2),(1,200),(1,400),(1,600),(2,3),(2,100),(2,500)))
    # test regular
    # x_in = np.array(((100,100),(200,150),(250,250),(150,300),(100,400),(50,400),(20,200),(400,200),(700,80),
    #                  (200,400),(250,600),(300,900),(400,1000),(500,800),(600,700),(500,300),(800,200),(100,900),
    #                  (400,1000),(20,900),(80,300),(200,600),(700,700),(550,750),(750,500)))
    # test (0,x)
    # x_in = np.array(((0,1),(0,2),(0,3),(0,5),(0,10),(0,20),(0,40),(0,50),(0,60),
    #                  (0,100),(0,200),(0,400),(0,500),(0,600),(0,800),(0,1000)))
    # # test small
    # x_in = np.array(((1,1),(1,2),(3,4),(2,5),(6,10),(7,1),(8,3),(10,2),(3,2),
    #                  (1,10),(2,9),(10,20),(20,9),(15,6),(7,8),(14,13)))
    # # test regular [0,300]
    # x_in = np.array(((10,20),(20,30),(50,80),(100,120),(150,200),(150,300),(200,300),(250,200),(280,180),
    #                  (200,80),(160,30),(150,60),(190,100),(110,10),(80,20),(10,50),(5,200),(200,5),
    #                  (90,45),(150,100),(120,5),(150,30),(300,100),(250,150),(160,20)))

    # # # Set 1 - low population
    A = np.array(((1,1,0),(0,1,1)))
    B = np.array(((2,0,0),(0,2,0)))
    c = np.array((0.2,0.02,0.2))
    # # # x_in = np.array(((5,5),(15,25),(25,40),(35,30),(45,40),(55,65),(65,70),(75,55),(85,60)))
    # x_in = np.array(((5,5),(15,25),(25,40),(35,30),(45,40),(55,65),(65,70),(75,55),(85,60),
    #                  (0,1),(0,5),(0,10),(0,20),(0,50),(0,70),(0,80),
    #                  (2,1),(3,5),(4,8),(1,2),(1,3),(2,4),(4,2),(3,4),(2,2)))
    # # test regular
    # x_in = np.array(((1,2),(3,4),(10,20),(10,30),(10,40),(20,50),(30,60),(70,20),(80,85),
    #                  (20,4),(5,60),(1,80),(80,1),(60,2),(7,40),(5,30),(10,2),(20,9),
    #                  (30,50),(20,70),(9,30),(15,25),(37,46),(4,50),(2,34)))
    # test (0,x)
    # x_in = np.array(((0,1),(0,2),(0,3),(0,5),(0,10),(0,20),(0,40),(0,50),(0,60),
    #                  (0,70),(0,80),(0,85),(0,45),(0,55),(0,65),(0,75)))
    # test regular [0,10]
    x_in = np.array(((1,1),(1,3),(1,5),(1,7),(1,9),(2,1),(2,4),(2,6),(2,8),
                     (2,10),(4,1),(4,3),(4,5),(4,7),(4,9),(5,2),(5,3),(5,5),
                     (5,6),(5,8),(8,1),(8,2),(8,10),(9,3),(10,10)))

    # Set 2 - Glispie
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((10,0.01,10))
    # x_in = np.array(((1000,1000),(2000,2000),(3000,3000),(4000,4000),(1500,2500),(2500,3000),(3000,1500),(500,500),(0,0)))
    
    data_dic = {}
    for j in range(x_in.shape[0]):
        print(x_in[j])
        x1d,x2d = x_in[j]
        xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data))).T
        data = ((GenDatawithGillespieSSA(A,B,c,xIC,t,longtime))[:,1,:]).T
        data_dic[str(j)+'_i'] = x_in[j]
        data_dic[str(j)+'_d'] = data
    data_dic['size'] = np.array((np.sqrt(x_in.shape[0]),np.sqrt(x_in.shape[0])))

    # K = 20
    # x_in = np.linspace(0,850,K).astype('int')
    # y_in = np.linspace(1,1050,K).astype('int')
    # xm,ym = np.meshgrid(x_in,y_in)
    # re = np.zeros(xm.shape)
    # re2 = np.zeros(xm.shape)
    # re3 = np.zeros(xm.shape)
    # re4 = np.zeros(xm.shape)

    # for j in range(x_in.shape[0]):
    #     for k in range(y_in.shape[0]):
    #         x1d,x2d = xm[j,k],ym[j,k]
    #         print('[%f,%f]'%(x1d,x2d))
    #         xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data))).T
    #         data = ((GenDatawithGillespieSSA(A,B,c,xIC,t,longtime))[:,1,:]).T
    #         # re[j] = np.std(data.flatten())/np.sqrt(T)
    #         # re2[j] = (np.mean(data.flatten())-x1d)/T
    #         re[j,k]  = (np.mean(data[:,0])-x1d)
    #         re2[j,k] = (np.mean(data[:,1])-x2d)
    #         re3[j,k] = np.std(data[:,0])
    #         re4[j,k] = np.std(data[:,1])
    # data_dic = {'x':xm,'y':ym,'m1':re,'m2':re2,'s1':re3,'s2':re4}

    return data_dic

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
    dim = 2

    # Set 1 - Stochastic modelling for systems biology, Wilkinson, Darren James, Fig 6.7
    A = np.array(((1,1,0),(0,1,1)))
    B = np.array(((2,0,0),(0,2,0)))
    c = np.array((1.0,0.005,0.6))
    x_in = np.array(((250,250),(300,300)))

    # # # Set 1 - low population
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((0.2,0.02,0.2))
    # x_in = np.array(((10,10),(20,20)))

    # Set 2 - Glispie
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((10,0.01,10))
    # x_in = np.array(((1000,1000),(2000,2000),(3000,3000),(4000,4000),(1500,2500),(2500,3000),(3000,1500),(500,500),(0,0)))
    
    data_dic = {}
    zerocond = lambda x: abs(x[0])<1.0e-8 or abs(x[1])<1.0e-8
    for j in range(x_in.shape[0]):
        data = []
        count = 0
        for i in range(N_data):
            print(i)
            data.append(GillespieSSA_stopptime(A,B,c,x_in[j],zerocond,1000))
        data_dic[str(j)+'_i'] = x_in[j]
        data_dic[str(j)+'_d'] = np.array(data)

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
    dim = 2

    # # Set 1 - Stochastic modelling for systems biology, Wilkinson, Darren James, Fig 6.7
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((1.0,0.005,0.6))
    # x1r,x2r = [0,650],[0,700]
    # x1d,x2d = 100,100

    # Set 1 - Stochastic modelling for systems biology, Wilkinson, Darren James, Fig 6.7 - long time T=100
    A = np.array(((1,1,0),(0,1,1)))
    B = np.array(((2,0,0),(0,2,0)))
    c = np.array((1.0,0.005,0.6))
    x1r,x2r = [0,850],[1,1050]
    x1d,x2d = 100,100

    # # Set 1 - low population
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((0.2,0.02,0.2))
    # x1r,x2r = [0,90],[0,90]
    # x1d,x2d = 10,10

    # # Set 2 - Glispie
    # A = np.array(((1,1,0),(0,1,1)))
    # B = np.array(((2,0,0),(0,2,0)))
    # c = np.array((10,0.01,10))
    # x1r,x2r = [0,5000],[0,5000]
    # x1d,x2d = 1000,1000
    
    # initial condition - can be changed
    xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data))).T
    data = GillespieSSAOriginal(A,B,c,xIC,t)
    return data

# def zeross(dat):
#     datNum = dat.shape[-1]
#     re = []
#     costraint = lambda x: (np.abs(x[:,0])<1.0e-8)+(np.abs(x[:,1])<1.0e-8)
#     for i in range(datNum):
#         kkk = np.where(costraint(dat[:,:,i].T))[0]
#         if len(kkk)>0:
#             re.append(np.min(kkk))
#         else:
#             pass
#     return re

# b = sio.loadmat('/Users/jesse/Dropbox/DataProd/Ex23SSALV_test.mat')
# b2 = sio.loadmat('/Users/jesse/Dropbox/SdeNF/results/Ex23_ResSDENF_s5_1/Test_SSAmodel/predict.mat')

# # histogram
# hist(zeross(b['data']),bins=100,histtype='step',range=(0,100))
# hist(zeross(b2['pred']),bins=100,histtype='step',range=(0,100))
# plt.show()

# for i in range(10):
#     for j in range(2):
#         plot(np.linspace(0,100,1001),b['data'][j,:,i])
#     plt.show()

# print('----------------------Next are predictions------------------------------')

# for i in range(10):
#     for j in range(2):
#         plot(np.linspace(0,100,1001),b2['pred'][j,:,i])
#     plt.show()

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    # data_train = Gendata(T=0.1,   Nt=1,      N_data=200000,  ifrandom=True,longtime=False)
    # sio.savemat(traindatapath,{'data':data_train})
    data_test  = Gendata(T=100.0,   Nt=1000,   N_data=50,  ifrandom=False,longtime=True)
    sio.savemat(testdatapath ,{'data':data_test})
    
    # conddatapath  = '../'+filename+'_cond.mat'
    # data_dic = Gendata_condition(T=0.1,   Nt=1,      N_data=10000,longtime=False)
    # sio.savemat(conddatapath,data_dic)

    oridatapath  = '../'+filename+'_ori.mat'
    data_dic = GendataOri(T=100.0,   Nt=1000,      N_data=10)
    sio.savemat(oridatapath,data_dic)

    # stoppingtimedatapath  = '../'+filename+'_stoppingtime.mat'
    # data_dic = Gendata_stoppingtime(T=0.1,   Nt=1,  N_data=10000)
    # sio.savemat(stoppingtimedatapath,data_dic)
