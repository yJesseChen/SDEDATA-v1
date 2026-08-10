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

# def geneq(dim,Delta,h):
#     sigma = 0.1
#     epsilon = 1.0
#     diagvec = np.zeros(dim)
#     for k in range(dim):
#         K = (k+1)//2
#         if k==0:
#             diagvec[k] = 0
#         else:
#             diagvec[k] = K**2
#     A = -np.pi*np.diag(diagvec)*epsilon
#     I = np.pi*np.eye(dim)
#     def drift(x):
#         return x@(A.T)
#     def diff(x):
#         return sigma*I
#     return drift,diff

def geneq(dim,Delta,h):
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
    def drift(x):
        return x@(A.T)
    def diff(x):
        return sigma*np.dot(M,B)
    return drift,diff

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
    ### initial condition
    if IC_=='value':
        # Fourier series for exp(-(sinx)^2)-1
        initial_f = lambda x: np.exp(-(np.sin(x))**2)-1
        initial_Ix = initial_f(Ix)
        modal_i = np.dot(iBasis,initial_Ix)
        initial = np.tile(modal_i,(N_data,1))
    # elif IC_=='uniform':
    #     initial = np.zeros([2*Npara+1,N_data])
    #     initial[[0]] = 1*np.random.rand(1,N_data)-0.5
    #     for n in np.arange(Npara)+1:
    #         initial[[2*n-1,2*n]] = 2.0/n*np.random.rand(2,N_data)-1.0/n
    elif IC_=='uniform':
        initial = np.zeros([Nx,N_data])
        xl,xr = ini_generate(Nx)
        for i in np.arange(Nx):
            initial[i,:] = (xr[i]-xl[i])*np.random.rand(N_data)+xl[i]
        initial = initial.T
    # ### generate in modal space
    # for k in range(2*Npara+1):
    #     K = (k+1)//2
    #     OneMat = np.tile(np.ones(N_data),(Nt+1,1))
    #     EXPMat = np.tile(np.exp(-c*(K**4)*t),(N_data,1)).T
    #     if k==0:
    #         data[0,:,:] = initial[0]*OneMat
    #     else:
    #         data[k,:,:] = initial[k]*EXPMat

    # function
    drift,diffu = geneq(Nx,T/Nt,h)
    # data
    data = np.zeros((Nx,Nt+1,N_data))
    datag = EM_auto_md_BE(drift,diffu,Nx,initial,t)
    for i in range(Nx):
        data[i,:,:] = (datag[:,i::Nx]).T
    if Nodal:
        data = np.einsum('ij,jkl', Basis, data)
    # datam = data

    # if steady:
    #     Nt = 1
    #     data = data[:,[0,-1],:]
    #     # data[0][1] -= np.mean(data[0][1])
    #     # data = np.tile(data,(10,1,1))
    if N_data==1:
        data = data.reshape([1,Nt+1])
    return data

def Gendata2D_withIC(Ix,h,T,Nt,N_data,IC,Nodal=False):
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
    ### initial condition
    initial = np.tile(IC,[N_data,1])
    # ### transfer modal data to nodal data
    # initial = np.dot(Basis,initial).T

    # function
    drift,diffu = geneq(Nx,T/Nt,h)
    # data
    data = np.zeros((Nx,Nt+1,N_data))
    datag = EM_auto_md_BE(drift,diffu,Nx,initial,t)
    for i in range(Nx):
        data[i,:,:] = (datag[:,i::Nx]).T
    if Nodal:
        data = np.einsum('ij,jkl', Basis, data)
    return data

def ini_generate(dim):
    xl = np.array((-0.8,-0.25,-0.2,-0.15,-0.1,-0.1,-0.1,-0.1,-0.1,-0.05))
    xr = np.array((-0.1, 0.25, 0.2, 0.4,  0.1, 0.1, 0.1, 0.1, 0.1, 0.05))
    if dim <= 10:
        xl,xr = xl[:dim],xr[:dim]
    else:
        xl = np.concatenate([xl,-0.05*np.ones(dim-10)])
        xr = np.concatenate([xr, 0.05*np.ones(dim-10)])
    return xl,xr

def Gendata_condition(Ix,h,T,Nt,N_data,Nodal=False):
    #
    # Generate conditional data
    # 
    Nx = Ix.shape[0]
    xl,xr = ini_generate(Nx)
    x_in = np.array((xl,
                     (xl+xr)/2,
                     1/4*xl+3/4*xr,
                     3/4*xl+1/4*xr
                    ))
    
    data_dic = {}
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
    # Generate
    for j in range(x_in.shape[0]):
        data = ((Gendata2D_withIC(Ix,h,T,Nt,N_data,x_in[j],Nodal))[:,1,:]).T
        if Nodal:
            data_dic[str(j)+'_i'] = np.dot(Basis,x_in[j])
        else:
            data_dic[str(j)+'_i'] = x_in[j]
        data_dic[str(j)+'_d'] = data
    data_dic['size'] = np.array((np.sqrt(x_in.shape[0]),np.sqrt(x_in.shape[0])))
    return data_dic

# Nx = a['data'].shape[0]
# Nd = 300
# I_X = np.linspace(0.0,2*np.pi,Nd)
# Basis = np.zeros([Nd,Nx])
# for k in range(Nx):
#     K = (k+1)//2
#     if k==0:
#         col = np.ones(Nd)
#     elif k%2==1:
#         col = np.cos(K*I_X)
#     elif k%2==0:
#         col = np.sin(K*I_X)
#     Basis[:,k] = col
# ndata = np.einsum('ij,jkl', Basis, a['data'][:,:,:4])

# I_X2 = np.linspace(0.0,2*np.pi,Nx+1)[:-1]
# Basis2 = np.zeros([Nx,Nx])
# for k in range(Nx):
#     K = (k+1)//2
#     if k==0:
#         col = np.ones(Nx)
#     elif k%2==1:
#         col = np.cos(K*I_X2)
#     elif k%2==0:
#         col = np.sin(K*I_X2)
#     Basis2[:,k] = col
# ndata2 = np.einsum('ij,jkl', Basis2, a['data'][:,:,:4])
# T = 150
# for j in range(4):
#     plot(I_X,ndata[:,T,j])
# plt.gca().set_prop_cycle(None)
# for j in range(4):
#     plot(I_X2,ndata2[:,T,j],linestyle='--')
# plt.gca().set_prop_cycle(None)
# for j in range(4):
#     scatter(I_X2,ndata2[:,T,j],s=30)

# def exp_to_sincos(ffts):
#     # input is the result of np.fft.fft
#     Nx = len(ffts)
#     fftss = np.fft.fftshift(ffts)
#     trigseq = np.zeros(Nx)
#     if Nx%2==0:
#         trigseq[0] = np.real(fftss[int(Nx/2)])
#         trigseq[1:-1:2] = 2*np.real(fftss[(int(Nx/2)+1):])
#         trigseq[2::2] = -2*np.imag(fftss[(int(Nx/2)+1):])
#         trigseq[-1] = np.real(fftss[0])
#     else:
#         trigseq[0] = np.real(fftss[int(Nx/2)])
#         trigseq[1:-1:2] = 2*np.real(fftss[(int(Nx/2)+1):])
#         trigseq[2::2] = -2*np.imag(fftss[(int(Nx/2)+1):])
#     return trigseq


if __name__ == '__main__':
    os.chdir(sys.path[0])
    filename = (sys.argv[0].split('/')[-1].split('.')[0])
    traindatapath = '../'+filename+'_train.mat'
    testdatapath  = '../'+filename+'_test.mat'
    I_X = np.linspace(0.0,2*np.pi,31)
    I_X = I_X[:-1]
    h_x = 2*np.pi/30
    data_train = Gendata2D(Ix=I_X,h=h_x,T=0.05, Nt=1, N_data=200000, IC_='uniform', Nodal=True)
    data_test  = Gendata2D(Ix=I_X,h=h_x,T=10.0, Nt=200, N_data=10000,  IC_='value', Nodal=True)
    sio.savemat(traindatapath,{'data':data_train})
    sio.savemat(testdatapath ,{'data':data_test})

    conddatapath  = '../'+filename+'_cond.mat'
    data_dic = Gendata_condition(Ix=I_X,h=h_x,T=0.05, Nt=1, N_data=1000, Nodal=True)
    sio.savemat(conddatapath,data_dic)
