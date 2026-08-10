import numpy as np
import sys
import os
import scipy
import scipy.io as sio
import pdb
import time
# import psutil
import gc


def exp_(c_,k_):
    # this is for f(t) = c*e^{kt}
    def tau_select(t,r):
        # find tau, such that int_{t}^{t+tau} f(s) ds = r
        return 1/k_*np.log(k_*r/(c_*np.exp(k_*t))+1)
    def inte_select(a,b):
        # compute int_{a}^{b} f(s) ds
        return c_/k_*(np.exp(k_*b)-np.exp(k_*a))
    return tau_select,inte_select

def psin_(A_,B_,o_):
    # this is for f(t) = A_+B_*sin(o_*t)
    def tau_select(t,r):
        # find tau, such that int_{t}^{t+tau} f(s) ds = r
        f = lambda tau: A_*tau-B_/o_*np.cos(o_*(t+tau))-r+B_/o_*np.cos(o_*t)
        fprime_ = lambda tau: A_+B_*np.sin(o_*(t+tau))
        fprime2_ = lambda tau: o_*B_*np.cos(o_*(t+tau))
        
        re = -1
        ini = 0
        while re<0: 
            try:
                re = scipy.optimize.newton(f, ini, fprime=fprime_, fprime2=fprime2_)
            except:
                re = -1
            ini +=1.0
            if ini>20:
                re = np.inf
        return re
    def inte_select(a,b):
        # compute int_{a}^{b} f(s) ds
        return A_*(b-a)+B_/o_*(np.cos(o_*a)-np.cos(o_*b))
    return tau_select,inte_select

def P2(G1,G2,G3):
    # this is for f(t) = G1+G2*t+G3*t**2
    def tau_select(t,r):
        # find tau, such that int_{t}^{t+tau} f(s) ds = r
        f = lambda tau: G3/3*tau**3+(t*G3+G2/2)*tau**2+(G1+t*G2+t**2*G3)*tau-r
        fprime_ = lambda tau: G3*tau**2+(2*t*G3+G2)*tau+(G1+t*G2+t**2*G3)
        fprime2_ = lambda tau: 2*G3*tau+2*t*G3+G2
        
        re = -1
        ini = 0
        while re<0: 
            try:
                re = scipy.optimize.newton(f, ini, fprime=fprime_, fprime2=fprime2_)
            except:
                re = -1
            ini +=1.0
            if ini>20:
                re = np.inf
        return re
    def inte_select(a,b):
        # compute int_{a}^{b} f(s) ds
        return G1*(b-a)+G2/2*(b**2-a**2)+G3/3*(b**3-a**3)
    return tau_select,inte_select

def NRMinfo(c,cinfo):
    if cinfo['func']=='exp':
        tau_select,inte_select = exp_(cinfo['func_para'][0],cinfo['func_para'][1])
    elif cinfo['func']=='psin':
        tau_select,inte_select = psin_(cinfo['func_para'][0],cinfo['func_para'][1],cinfo['func_para'][2])
    elif cinfo['func']=='polynomial2':
        tau_select,inte_select = P2(cinfo['func_para'][0],cinfo['func_para'][1],cinfo['func_para'][2])
    else:
        raise AttributeError('Not supported')

    # info for c = (k(t),k2,k3,k4)
    def harzard(x):
        return np.array((1,x[0],x[0],x[1]))
    def tauf(x,t,Ts,Ss):
        re = np.zeros(Ts.shape)
        h = harzard(x)
        # constant rate
        re[1:] = (Ss[1:]-Ts[1:])/(h[1:]*c[1:])
        # time-dependent
        re[0]  = tau_select(t,(Ss[0]-Ts[0])/h[0])
        return re
    def inteaf(x,a,b):
        h = harzard(x)
        re = np.zeros(h.shape)
        # constant rate
        re[1:] = h[1:]*c[1:]*(b-a)
        # time-dependent
        re[0]  = h[0]*inte_select(a,b)
        return re
    return tauf,inteaf

def ModifiedNextReactionMethod(A,B,c,cinfo,initial,T):
    # Shape: initial : [1,         N_species]
    #        A,B     : [N_species, N_reaction]
    #        c       : [1,         N_reaction]
    #        cinfo   : dictionary with information of the reactions
    N_species = initial.shape[0]
    N_reactions = A.shape[1]
    S = B-A
    react_time = [0]
    react_numb = [initial]
    tauf,inteaf = NRMinfo(c,cinfo)
    x = initial
    t = 0
    Ts = np.zeros(N_reactions)
    Ss = np.log(1/np.random.uniform(0,1,N_reactions))
    while t<T:
        taus = tauf(x,t,Ts,Ss)
        j = np.argmin(taus)
        tau  = taus[j]
        Ts = Ts+inteaf(x,t,t+tau)
        x = x + S[:,j]
        if tau<0:
            pdb.set_trace()
        t = t+tau
        Ss[j] += np.log(1/np.random.uniform(0,1))
        react_time.append(t)
        react_numb.append(x)
    react_time = np.array(react_time)
    react_numb = np.vstack(react_numb)
    return react_time,react_numb

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

def info_to_func(cinfo):
    if cinfo['func']=='exp':
        c_,k_ = cinfo['func_para'][0],cinfo['func_para'][1]
        return lambda t: c_*np.exp(k_*t)
    elif cinfo['func']=='psin':
        A_,B_,o_ = cinfo['func_para'][0],cinfo['func_para'][1],cinfo['func_para'][2]
        return lambda t: A_+B_*np.sin(o_*t)
    elif cinfo['func']=='polynomial2':
        G1,G2,G3 = cinfo['func_para'][0],cinfo['func_para'][1],cinfo['func_para'][2]
        return lambda t: G1+G2*t+G3*t**3
    else:
        raise AttributeError('Not supported')

def sample_pos_poly(Nt,N_data,dt,dat_r):
    re = np.zeros([3,Nt,N_data])
    lenid1 = 0
    while lenid1<N_data:
        para_data = np.array((np.random.uniform(dat_r[0][0],dat_r[0][1],[Nt,N_data]),np.random.uniform(dat_r[1][0],dat_r[1][1],[Nt,N_data]),np.random.uniform(dat_r[2][0],dat_r[2][1],[Nt,N_data])))
        q = np.squeeze(para_data).T
        # check if positive polynomial
        cond1 = (q[:,2]>0)*(q[:,1]<0)*((q[:,1]+2*dt*q[:,2])>0)
        check1 = gamma2(dt,q[:,0],q[:,1],q[:,2])>0
        check2 = q[:,1]**2<4*q[:,0]*q[:,2]
        check = (~cond1)*check1 + cond1*check2
        id_ = np.where(check)[0]

        lenid = min(len(id_)+lenid1,N_data)
        re[:,:,lenid1:lenid] = (para_data[:,:,id_])[:,:,:(lenid-lenid1)]
        lenid1 = lenid
    return re

def GenDatawithGillespieSSA(A,B,c,cinfo,initial,t_step,clist=None):
    dim = initial.shape[1]
    N_data = initial.shape[0]
    data_re = np.zeros([dim,t_step.shape[0],N_data])
    for i in range(N_data):
        st = time.time()
        print(i)
        if clist is not None:
            cinfo['func_para'] = clist[i]
        react_time,react_numb = ModifiedNextReactionMethod(A,B,c,cinfo,initial[i],t_step[-1])
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

def GillespieSSAOriginal(A,B,c,cinfo,initial,t_step):
    dim = initial.shape[1]
    N_data = initial.shape[0]
    data_re = {}
    for i in range(N_data):
        st = time.time()
        print(i)
        react_time,react_numb = ModifiedNextReactionMethod(A,B,c,cinfo,initial[i],t_step[-1])
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
    dim = 2

    # Set 2 - Glispie
    A = np.array(((0,1,1,0),(0,0,0,1)))
    B = np.array(((1,1,0,0),(0,1,0,0)))
    c = np.array((-1,500,20.0,5.0))
    # cinfo = {'func':'exp','func_para':[0.5,-1.0]}
    
    x1r,x2r = [0,10],[0,400]
    x1d,x2d = 1,1
    
    # initial condition - can be changed
    if ifrandom:
        cinfo = {'func':'polynomial2','func_para':[1,1,1]}

        perc = [0.3,0.2,0.15,0.35]
        Nperc = [int(perc[i]*N_data) for i in range(4)]
        Nperc[3] = N_data-sum(Nperc[:-1])
        
        xIC1 = np.array((np.maximum(np.random.randint(x1r[0],x1r[1],Nperc[0]),0),np.maximum(np.random.randint(x2r[0],x2r[1],Nperc[0]),0))).T
        para_data1 = sample_pos_poly(Nt,Nperc[0],t[1]-t[0],dat_r=[[0,40],[-5.3,5.3],[-0.7,0.7]])
        
        # bd data treatment
        xIC2 = np.array((np.maximum(np.random.randint(0,1,Nperc[1]),0),np.maximum(np.random.randint(0,300,Nperc[1]),0))).T
        para_data2 = sample_pos_poly(Nt,Nperc[1],t[1]-t[0],dat_r=[[0,40],[-5.3,5.3],[-0.7,0.7]])

        xIC3 = np.array((np.maximum(np.random.randint(0,5,Nperc[2]),0),np.maximum(np.random.randint(0,1,Nperc[2]),0))).T
        para_data3 = sample_pos_poly(Nt,Nperc[2],t[1]-t[0],dat_r=[[0,10],[-5.3,5.3],[-0.7,0.7]])

        xIC4 = np.array((np.maximum(np.random.randint(0,1,Nperc[3]),0),np.maximum(np.random.randint(0,1,Nperc[3]),0))).T
        para_data4 = sample_pos_poly(Nt,Nperc[3],t[1]-t[0],dat_r=[[0,10],[-5.3,5.3],[-0.7,0.7]])
        

        xIC = np.concatenate([xIC1,xIC2,xIC3,xIC4],axis=0)
        para_data = np.concatenate([para_data1,para_data2,para_data3,para_data4],axis=-1)
        idx = np.arange(N_data)
        np.random.shuffle(idx)
        xIC,para_data = xIC[idx,:],para_data[:,:,idx]

        data = GenDatawithGillespieSSA(A,B,c,cinfo,xIC,t,clist=np.squeeze(para_data).T)
    else:
        cinfo = {'func':'psin','func_para':[20.0,20.0,2*np.pi/24]}
        xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data))).T
        para_data = Gamma2(t,info_to_func(cinfo))
        data = GenDatawithGillespieSSA(A,B,c,cinfo,xIC,t)
    # data
    if N_data==1:
        data      = data.reshape([1,Nt+1])
        para_data = para_data.reshape([dim_para,Nt])
    return data,para_data

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
    dim = 2

    A = np.array(((0,1,1,0),(0,0,0,1)))
    B = np.array(((1,1,0,0),(0,1,0,0)))
    c = np.array((-1,500,20.0,5.0))
    cinfo = {'func':'polynomial2','func_para':[1,1,1]}
    x_in = np.array(((1, 1),(2,174),(5,287),(5,219),(2,133),(1,161),(0,19),(0,1),(0,0)))
    p_in = np.array(((2.00000000e+01,  5.23628680e+00, -8.97133926e-03),
                     (32.17522858,  4.15422383, -0.42431424),
                     (39.31851653,  1.35524791, -0.66429091),
                     (38.47759065, -2.00384292, -0.62972058),
                     (30.,         -4.53475885, -0.33489094),
                     (17.38947616, -5.19148925,  0.09834689),
                     (4.13293319, -3.1876471,   0.54916286),
                     (0.68148347, -1.35524791,  0.66429091),
                     (0.17110277, -0.68346967,  0.68062867)))
    
    data_dic = {}
    for j in range(x_in.shape[0]):
        x1d,x2d = x_in[j]
        xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data))).T
        cinfo['func_para'] = p_in[j]
        data = ((GenDatawithGillespieSSA(A,B,c,cinfo,xIC,t))[:,1,:]).T
        data_dic[str(j)+'_i'] = x_in[j]
        data_dic['para_'+str(j)+'_i'] = p_in[j]
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
    dim = 2

    A = np.array(((0,1,1,0),(0,0,0,1)))
    B = np.array(((1,1,0,0),(0,1,0,0)))
    c = np.array((-1,500,20.0,5.0))
    cinfo = {'func':'psin','func_para':[20.0,20.0,2*np.pi/24]}
    x1d,x2d = 1,1
    
    # initial condition - can be changed
    xIC = np.array((x1d*np.ones(N_data,dtype='int'),x2d*np.ones(N_data))).T
    data = GillespieSSAOriginal(A,B,c,cinfo,xIC,t)
    return data

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    data_train, para_train = Gendata(T=0.1,   Nt=1,      N_data=200000,  ifrandom=True)
    sio.savemat(traindatapath,{'data':data_train,'para':para_train})
    # data_test , para_test  = Gendata(T=120.0,   Nt=1200,   N_data=1000,  ifrandom=False)
    # sio.savemat(testdatapath ,{'data':data_test ,'para':para_test })
    
    # conddatapath  = '../'+filename+'_cond.mat'
    # data_dic = Gendata_condition(T=0.1,   Nt=1,      N_data=500)
    # sio.savemat(conddatapath,data_dic)

    # oridatapath  = '../'+filename+'_ori.mat'
    # data_dic = GendataOri(T=240.0,   Nt=3,      N_data=10)
    # sio.savemat(oridatapath,data_dic)
