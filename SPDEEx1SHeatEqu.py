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

def EM_auto_md_BE(drift,diffusion,dim,initial,t_steps):
    data = np.zeros([initial.shape[0],dim*t_steps.shape[0]])
    data[:,:dim] = initial
    noise = std_normal(initial.shape[0], t_steps-1, dim)
    diff = t_steps[1:]-t_steps[:-1]
    for i in range(t_steps.shape[0]-1):
        Xt = data[:,i*dim:(i+1)*dim]
        data[:,(i+1)*dim:(i+2)*dim] = drift(Xt)+((noise[:,(i+1)*dim:(i+2)*dim][:,None,:])@(diffusion(Xt)))[:,0,:]
    return data

def geneq(dim,Delta,h):
    sigma = 0.1
    epsilon = 1.0
    A = (np.diag(np.full(dim,2))-np.diag(np.ones(dim-1),1)-np.diag(np.ones(dim-1),-1))/h**2
    M = np.linalg.inv(np.eye(dim)+Delta*epsilon*A)
    def drift(x):
        return x@(M.T)
    def diff(x):
        return sigma*M
    return drift,diff

def Gendata2D(Npara,Ix,h,T,Nt,N_data,IC_):
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
    Nx = Ix.shape[0]
    t = np.linspace(0,T,Nt+1)
    data = np.zeros((2*Npara+1,Nt+1,N_data))
    ### initial condition
    if IC_=='value':
        # Fourier series for exp(-(sinx)^2)-1
        modal_i = -np.array((0.14503527044915-0.5,0,0,0.31284160636974334,0,0,0,0.03870411541932656,0,0,0,0.0032086830151304125,0,0,0))
        initial = np.tile(modal_i,(N_data,1)).T
    # elif IC_=='uniform':
    #     initial = np.zeros([2*Npara+1,N_data])
    #     initial[[0]] = 1*np.random.rand(1,N_data)-0.5
    #     for n in np.arange(Npara)+1:
    #         initial[[2*n-1,2*n]] = 2.0/n*np.random.rand(2,N_data)-1.0/n
    elif IC_=='uniform':
        initial = np.zeros([2*Npara+1,N_data])
        for n in np.arange(Npara)+1:
            initial[[2*n]] = 2.0/n*np.random.rand(1,N_data)-1.0/n
    # ### generate in modal space
    # for k in range(2*Npara+1):
    #     K = (k+1)//2
    #     OneMat = np.tile(np.ones(N_data),(Nt+1,1))
    #     EXPMat = np.tile(np.exp(-c*(K**4)*t),(N_data,1)).T
    #     if k==0:
    #         data[0,:,:] = initial[0]*OneMat
    #     else:
    #         data[k,:,:] = initial[k]*EXPMat
    ### transfer modal data to nodal data
    Basis = np.zeros([Nx,2*Npara+1])
    for k in range(2*Npara+1):
        K = (k+1)//2
        if k==0:
            col = np.ones(Nx)
        elif k%2==1:
            col = np.cos(K*Ix)
        elif k%2==0:
            col = np.sin(K*Ix)
        Basis[:,k] = col
    initial = np.dot(Basis,initial).T

    # function
    drift,diffu = geneq(Nx,T/Nt,h)
    # data
    data = np.zeros((Nx,Nt+1,N_data))
    datag = EM_auto_md_BE(drift,diffu,Nx,initial,t)
    for i in range(Nx):
        data[i,:,:] = (datag[:,i::Nx]).T
    # if steady:
    #     Nt = 1
    #     data = data[:,[0,-1],:]
    #     # data[0][1] -= np.mean(data[0][1])
    #     # data = np.tile(data,(10,1,1))
    if N_data==1:
        data = data.reshape([1,Nt+1])
    return data

if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    I_X = np.linspace(0.0,np.pi,21)
    I_X = I_X[1:-1]
    h_x = 2*np.pi/20
    data_train = Gendata2D(Npara=7,Ix=I_X,h=h_x,T=0.01, Nt=1, N_data=200000, IC_='uniform')
    data_test  = Gendata2D(Npara=7,Ix=I_X,h=h_x,T=2.0, Nt=200, N_data=5000,  IC_='value')
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})
