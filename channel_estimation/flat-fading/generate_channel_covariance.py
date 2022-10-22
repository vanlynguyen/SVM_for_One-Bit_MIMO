# Written by Van Ly Nguyen, WiiLab, San Diego State University (SDSU)
# Contact: https://sites.google.com/view/vanlynguyen/home

import numpy as np
import scipy.integrate as integrate
from scipy.linalg import toeplitz
from scipy.linalg import cholesky

N = 32

sqrt2 = np.sqrt(2)
pi = np.pi

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

filename1 = './Covariance_Data/Cov_mat_'+str(N)+'N.txt'
filename2 = './Covariance_Data/Lower_mat_'+str(N)+'N.txt'

print(np.random.uniform(size=4))

M = 100
sigma = 10*2*pi/360 # angular spread = 10 degrees
theta_min = -pi/2
theta_max =  pi/2
Sk_denom = 1/(sqrt2*sigma*(1-np.exp(-sqrt2*np.pi/sigma)))

k = 0
while k<M:
    theta_mean = (pi/3)*(2*np.random.uniform()-1)
    Sk = lambda theta: np.exp(-sqrt2*np.abs(theta-theta_mean)/sigma)/Sk_denom
    normalized_factor, _ = integrate.quad(Sk,theta_min,theta_max)
    first_column = np.zeros((N),dtype=complex)
    for n in range(N):
        R_kn_real = lambda theta: np.cos(pi*n*np.sin(theta))* \
            np.exp(-sqrt2*np.abs(theta-theta_mean)/sigma)/Sk_denom
        res_real, _ = integrate.quad(R_kn_real, theta_min, theta_max)
        
        R_kn_imag = lambda theta: np.sin(pi*n*np.sin(theta))* \
            np.exp(-sqrt2*np.abs(theta-theta_mean)/sigma)/Sk_denom
        res_imag, _ = integrate.quad(R_kn_imag, theta_min, theta_max)
        
        first_column[n] = (res_real - 1j*res_imag)/normalized_factor
    R_kn = toeplitz(first_column)
    if is_pos_def(R_kn)==True:
        k = k + 1
        print(k)
        print(theta_mean)
        print('---')
        #R_kn_inv = np.linalg.inv(R_kn)
        L = cholesky(R_kn,lower=True)
        
        f1 = open(filename1,'a')
        R_kn_real = np.concatenate((np.real(R_kn),np.imag(R_kn)),axis=1)
        np.savetxt(f1, R_kn_real, fmt='%.15f', delimiter=', ') # save channel list to a text file
        f1.close()
        
        f2 = open(filename2,'a')
        L_real = np.concatenate((np.real(L),np.imag(L)),axis=1)
        np.savetxt(f2, L_real, fmt='%.15f', delimiter=', ') # save channel list to a text file
        f2.close()



Cov_inv = np.genfromtxt(filename1, delimiter=', ')
Cov_inv = Cov_inv[:,0:N] + 1j*Cov_inv[:,N:2*N]

L_mat = np.genfromtxt(filename2, delimiter=', ')
L_mat = L_mat[:,0:N] + 1j*L_mat[:,N:2*N]
    
    
    