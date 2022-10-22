# Written by Van Ly Nguyen, WiiLab, San Diego State University (SDSU)
# Contact: https://sites.google.com/view/vanlynguyen/home

from scipy.linalg import dft
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import random

random.seed(100)

K  = 4   # number of users
N  = 32  # number of antennas at the base station
Tt = 20  # length of the training squence

min_snr  = -15   # minimum simulated SNR
max_snr  = 30    # maximum simulated SNR
snr_step = 5.0   # SNR step
snr_dB = np.arange(min_snr, max_snr+snr_step, snr_step) # snr in dB
snr = 10**(snr_dB/10)                          # snr in linear scale

NMSE = np.zeros(snr.shape)    # array to store normalized mean-squared-error

DFT_matrix = dft(Tt)          # DFT matrix of size Tt
Theta = DFT_matrix[:,1:K+1].T # training sequence contains K columns from 
                              # the 2nd to the (K+1)th columns of DFT_matrix

# Convert Theta to the real domain
X = np.concatenate((np.concatenate((np.real(Theta), np.imag(Theta)),1), \
                    np.concatenate((-np.imag(Theta),np.real(Theta)),1)),0)

clf = svm.LinearSVC(C = 1, loss='squared_hinge', fit_intercept=0, max_iter=30)

max_run_sim = 1e3

for jj in range(snr.shape[0]):
    print(snr_dB[jj])
    N0 = 1/snr[jj]

    for ii in range(np.int(max_run_sim)):
        
        H = np.random.normal(0,np.sqrt(0.5),(N,2*K))
        
        Noise = np.random.normal(0,np.sqrt(0.5*N0),(N,2*Tt)) # noise in real domain
        
        R = np.matmul(H,X)+Noise
        Y = (R>=0.0)+0 # classes of received signals: 0 and 1
        
        Hhat = np.zeros(H.shape) # initialize estimated channel matrix
        
        for n in range(N):
            clf.fit(X.T, Y[n,:])
            Hhat[n,:] = (np.sqrt(K)/np.linalg.norm(clf.coef_))*clf.coef_
        chanEstError = (np.linalg.norm(H-Hhat)**2)/(N*K)
        
        NMSE[jj] = NMSE[jj] + chanEstError
        
    NMSE[jj] = 10*np.log10(NMSE[jj]/max_run_sim)


plt.plot(snr_dB,NMSE,"-o")
plt.xlabel('SNR - dB')
plt.ylabel('NMSE - dB')
plt.grid(b=1,which='major',axis='both')
plt.axis([min_snr, max_snr, np.min(NMSE), np.max(NMSE)])
print(NMSE)
    
    