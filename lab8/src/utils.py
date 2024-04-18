import numpy as np
from matplotlib import pyplot as plt
import sys
from PIL import Image
import sympy
import os
import scipy
from scipy.ndimage import convolve

def threshold(im,T):
    c=im.shape[0]
    r=im.shape[1]
    tim=np.zeros([im.shape[0], im.shape[1]])
    for m in range(c):
        for n in range(r):
            if im[m][n]>T:
                tim[m][n]=255
            else:
                tim[m][n]=0
    return tim


def RMSE(tim, im):
    c=im.shape[0]
    r=im.shape[1]
    if tim.dtype==np.uint8 or im.dtype==np.uint8:
        f=im.clip(0,255).astype(np.double)
        b=tim.clip(0,255).astype(np.double)
    else:
        f=im
        b=tim
    return np.sqrt(np.square(f-b).sum()/(r*c))

def fidelity(tim, im):
    tg=ungamma(tim, 2.2)
    img=ungamma(im, 2.2)
    kernel=kernel_gauss(7, 2)

    # lt = filter_fir(tg, kernel)
    # lim = filter_fir(img, kernel)
    lt = convolve(tg, kernel, mode='nearest')
    lim = convolve(img, kernel, mode='nearest')
    
    btilde=255*pow((lt/255), (1/3))
    ftilde=255*pow((lim/255), (1/3))
    return RMSE(btilde, ftilde)

def ungamma(im, gamma):
    corrected=np.zeros([im.shape[0],im.shape[1]])
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            corrected[i][j]=255*pow((im[i][j]/255), gamma)
    return corrected

def kernel_gauss(ksize, sigma):
    lpfdx=ksize//2
    sq=np.square(range(-lpfdx, lpfdx+1)).reshape([ksize, 1])
    sq=sq+sq.T
    kernel=np.exp(-sq/(2*sigma))
    kernel/=kernel.sum()
    return kernel

def filter_fir(im, kernel):
    ky=kernel.shape[0]
    kx=kernel.shape[1]
    dy=ky//2
    dx=kx//2
    im_out=np.zeros([im.shape[0], im.shape[1]])
    im_out[:dy, :]=im[:dy, :]
    im_out[-dy:, :]=im[-dy:,:]
    im_out[:,:dx]=im[:,:dx]
    im_out[:,:-dx]=im[:,:-dx]
    for i in range(dy, im.shape[0]-dy):
        for j in range(dx, im.shape[1]-dx):
            im_out[i][j]=(im[i-dy:i-dy+ky, j-dx:j-dx+kx]*kernel).sum()
    return im_out

def idx_mat(size):
    I2=[[1,2],[3,0]]
    k=4*np.ones((2,2), dtype=int)
    I=0
    for j in range(size.bit_length()-1):
        I=np.kron(k,I)+np.kron(I2, np.ones_like(I, dtype=int))
    return I

def dither(x, N):
    H, W=x.shape
    print(x.shape)
    b=np.zeros_like(x, dtype=np.uint8)
    print(idx_mat(N))
    T=255*((idx_mat(N)+0.5)/(N*N))
    for i in range(0, H, N):
        di=min(H-i, N)
        for j in range(0, W, N):
            dj=min(W-j, N)
            # for m in range(di):
            #     for n in range(dj):
            #         if x[i+m][j+n]>T[]
            #         b[i+m][j+n]=255
            b[i:i+di, j:j+dj]=255*(x[i:i+dj, j:j+dj]>T[:di, :dj])
    return b
    
    
