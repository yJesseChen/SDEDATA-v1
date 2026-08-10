import numpy as np
import sys
import os
import scipy
import scipy.io as sio
import pdb

def std_normal(N_data, t_steps, dim):
    # x0 and t_steps should be 1d array
    diff = t_steps[1:]-t_steps[:-1]
    grow = np.zeros([N_data,t_steps.shape[0]*dim])
    for i in range(t_steps.shape[0]-1):
        grow[:,(i+1)*dim:(i+2)*dim] = np.random.normal(0.0, np.sqrt(diff[i]), [N_data,dim])
    return grow

def EM_md_BE(drift,diffusion,dim,initial,t_steps):
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = t_steps[1:]-t_steps[:-1]
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i*dim:(i+1)*dim]
        data[:,(i+1)*dim:(i+2)*dim] = drift(Xt,t_steps[i+1])+((noise[:,(i+1)*dim:(i+2)*dim][:,None,:])@(diffusion(Xt,t_steps[i])))[:,0,:]
    return data

def EMgamma_md_BE(drift,diffusion,dim,initial,para,t_steps):
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = t_steps[1:]-t_steps[:-1]
    dt = diff[0]
    b_ = gamma2(dt,para[0].T,para[1].T,para[2].T)
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i*dim:(i+1)*dim]
        data[:,(i+1)*dim:(i+2)*dim] = drift(Xt,b_[:,[i]])+((noise[:,(i+1)*dim:(i+2)*dim][:,None,:])@(diffusion(Xt,b_[:,[i]])))[:,0,:]
    return data

def ini_generate(dim):
    xl = np.array((-0.6,-0.2,-0.3,-0.2,-0.25,-0.1,-0.1,-0.1,-0.1,-0.05))
    xr = np.array((0.1,  0.3, 0.4, 0.4, 0.25, 0.1, 0.1, 0.1, 0.1, 0.05))
    if dim <= 10:
        xl,xr = xl[:dim],xr[:dim]
    else:
        xl = np.concatenate([xl,-0.05*np.ones(dim-10)])
        xr = np.concatenate([xr, 0.05*np.ones(dim-10)])
    return xl,xr

def fgen(s1,s2):
    def b(t):
        return np.sin(t/3)+np.cos(t)+2
    def f_exact(x,t):
        return np.array((x[:,0]-x[:,0]*x[:,1]+b(t),-x[:,1]+x[:,0]*x[:,1])).T
    def f_appx(x,b_):
        return np.array((x[:,0]-x[:,0]*x[:,1]+b_,-x[:,1]+x[:,0]*x[:,1])).T
    # def diff_exact(x,t):
    #     sigma = np.array(((s1,0),(0,s2)))
    #     return np.repeat(sigma[None,:,:],x.shape[0],axis=0)
    # def diff_appx(x,c_):
    #     sigma = np.array(((s1,0),(0,s2)))
    #     return np.repeat(sigma[None,:,:],x.shape[0],axis=0)
    def diff_exact(x,t):
        re = np.zeros([x.shape[0],2,2])
        re[:,0,0] = s1*x[:,0]
        re[:,1,1] = s2*x[:,1]
        return re
    def diff_appx(x,c_):
        re = np.zeros([x.shape[0],2,2])
        re[:,0,0] = s1*x[:,0]
        re[:,1,1] = s2*x[:,1]
        return re
    return f_exact,f_appx,b,diff_exact,diff_appx

def geneq(dim,Delta,h,p,q):
    sigma = 0.05
    epsilon = 0.1
    diagvec = np.zeros(dim)
    for k in range(dim):
        K = (k+1)//2
        if k==0:
            diagvec[k] = 0
        else:
            diagvec[k] = K**2
    I = np.eye(dim)
    M = np.linalg.inv(I+Delta*np.diag(diagvec)*epsilon)
    B = np.eye(dim)/np.sqrt(np.pi)
    B[0,0] = 1/np.sqrt(2*np.pi)
    A = M
    S = np.dot(M,Compute_S(dim,p,q))*Delta
    def alpha(t):
        return np.sin(3*t)
    def drift(x,t):
        return x@(A.T)+alpha(t)*S
    def diff(x,t):
        return sigma*np.dot(M,B)
    def drift_appx(x,a_):
        return x@(A.T)+a_*S
    def diff_appx(x,c_):
        return sigma*np.dot(M,B)
    return drift,diff,drift_appx,diff_appx,alpha

def Compute_S(n,p,q):
    re = np.zeros(n)
    for k in range(n):
        K = (k+1)//2
        if k==0:
            I = scipy.integrate.quad(lambda x: np.exp(-(x-p)**2/q**2), 0, 2*np.pi)[0]/(2*np.pi)
        else:
            if k%2==1:
                I = scipy.integrate.quad(lambda x: np.exp(-(x-p)**2/q**2)*np.cos(K*x), 0, 2*np.pi)[0]/np.pi
            else:
                I = scipy.integrate.quad(lambda x: np.exp(-(x-p)**2/q**2)*np.sin(K*x), 0, 2*np.pi)[0]/np.pi
        re[k] = I
    return re

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

## Taylor polynomial
# def Gamma2(tsq,b):
#     # given tseq t0, t1, t2,...
#     # solve Gamma0, Gamma1,... of local approximation wpt b
#     numT = tsq.shape[0]
#     dt = tsq[1]-tsq[0]
#     re = np.zeros([3,numT-1])
#     V  = np.vander(np.array((0,dt/2,dt)), increasing=True)
#     Vi = np.linalg.inv(V)
#     for i in range(numT-1):
#         re[0,i] = b(tsq[i])
#         re[1,i] = bp(tsq[i])
#         re[2,i] = bpp(tsq[i])/2
#     return re

# def bp(t):
#     return np.cos(t/3)/3-np.sin(t)

# def bpp(t):
#     return -np.sin(t/3)/9-np.cos(t)

## Legendre polynomial
# def Gamma2(tsq,b):
#     # given tseq t0, t1, t2,...
#     # solve Gamma0, Gamma1,... of local approximation wpt b
#     numT = tsq.shape[0]
#     dt = tsq[1]-tsq[0]
#     re = np.zeros([3,numT-1])
#     pts = np.array(((0,dt/2,dt),))
#     V  = Vander_L(pts, pts.shape[-1])
#     Vi = np.linalg.inv(V)
#     for i in range(numT-1):
#         re[:,i] = np.dot(Vi,b(np.array((tsq[i],tsq[i]+dt/2,tsq[i]+dt))))
#     pdb.set_trace()
#     return re

# def Vander_L(points,n):
#     Vander = []
#     for i in range(n):
#         lpoly = scipy.special.legendre(i)
#         Vander.append(lpoly(points))
#     Vander = np.concatenate(Vander,axis=0)
#     return Vander

def Gendata2D(Ix,h,T,Nt,N_data,IC_,Nodal=False):
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
    p = 1.0
    q = 1.0
    Nx = Ix.shape[0]
    # Npara = int((Nx-1)/2)
    t = np.linspace(0,T,Nt+1)
    data = np.zeros((Nx,Nt+1,N_data))
    # modal matrix
    Basis = np.zeros([Nx,Nx])
    for k in range(Nx):
        K = (k+1)//2
        if k==0:
            col = np.ones(Nx)
        elif k%2==1:
            col = np.cos(K*Ix)
        elif k%2==0:
            col = np.sin(K*Ix)
        Basis[:,k] = col
    iBasis = np.linalg.inv(Basis)

    # function
    drift,diffu,drift_appx,diffu_appx,alpha = geneq(Nx,T/Nt,h,p,q)
    # data
    dim_para = 3
    data = np.zeros((Nx,Nt+1,N_data))

    ### initial condition
    if IC_=='value':
        # Fourier series for exp(-(sinx)^2)-1
        initial_f = lambda x: np.exp(-(np.sin(x))**2)-1
        initial_Ix = initial_f(Ix)
        modal_i = np.dot(iBasis,initial_Ix)
        initial = np.tile(modal_i,(N_data,1))
        datag = EM_md_BE(drift,diffu,Nx,initial,t)
        para_data = Gamma2(t,alpha)
    elif IC_=='uniform':
        initial = np.zeros([Nx,N_data])
        xl,xr = ini_generate(Nx)
        for i in np.arange(Nx):
            initial[i,:] = (xr[i]-xl[i])*np.random.rand(N_data)+xl[i]
        initial = initial.T

        para_data = np.array((np.random.uniform(-1.2,1.2,[Nt,N_data]),np.random.uniform(-3.5,3.5,[Nt,N_data]),np.random.uniform(-5.0,5.0,[Nt,N_data])))
        datag = EMgamma_md_BE(drift_appx,diffu_appx,Nx,initial,para_data,t)

    for i in range(Nx):
        data[i,:,:] = (datag[:,i::Nx]).T
    if Nodal:
        data = np.einsum('ij,jkl', Basis, data)

    # if steady:
    #     Nt = 1
    #     data = data[:,[0,-1],:]
    #     # data[0][1] -= np.mean(data[0][1])
    #     # data = np.tile(data,(10,1,1))
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
    I_X = np.linspace(0.0,2*np.pi,31)
    I_X = I_X[:-1]
    h_x = 2*np.pi/30
    data_train, para_train = Gendata2D(Ix=I_X,h=h_x,T=0.05, Nt=1, N_data=200000, IC_='uniform', Nodal=False)
    data_test , para_test  = Gendata2D(Ix=I_X,h=h_x,T=10.0, Nt=200, N_data=10000,  IC_='value', Nodal=False)
    sio.savemat(traindatapath,{'data':data_train,'para':para_train})
    sio.savemat(testdatapath ,{'data':data_test ,'para':para_test })
