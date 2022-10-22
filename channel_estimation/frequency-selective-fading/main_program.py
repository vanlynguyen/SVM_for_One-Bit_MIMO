#from scipy.linalg import dft
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import random

random.seed(100)

K  = 2   # number of users
N  = 16  # number of antennas at the base station

min_snr  = -15   # minimum simulated SNR
max_snr  = 30    # maximum simulated SNR
snr_step = 5.0   # SNR step
snr_dB = np.arange(min_snr, max_snr+snr_step, snr_step) # snr in dB
snr = 10**(snr_dB/10)                          # snr in linear scale

NMSE = np.zeros(snr.shape)    # array to store normalized mean-squared-error

real_Theta_L = np.loadtxt("real_Theta_L.txt", dtype='double', delimiter=',')
imag_Theta_L = np.loadtxt("imag_Theta_L.txt", dtype='double', delimiter=',')
Tt = np.loadtxt("Np.txt", dtype='int') # length of the training sequence
L = np.loadtxt("L.txt", dtype='int')   # number of taps

# Convert Theta to the real domain
X = np.concatenate((np.concatenate((real_Theta_L, -imag_Theta_L),1), \
                    np.concatenate((imag_Theta_L, real_Theta_L),1)),0)

clf = svm.LinearSVC(C = 1, fit_intercept=0, max_iter=100)
max_run_sim = 1e2
for jj in range(snr.shape[0]):
    print(snr_dB[jj])
    N0 = 1/snr[jj]
    for ii in range(np.int(max_run_sim)):
        H = np.zeros((N,2*L*K))
        Hhat = np.zeros((N,2*L*K))
        for n in range(N):
            h_real = np.random.normal(0,np.sqrt(0.5/L),(L*K,1)) # real part of channel
            h_imag = np.random.normal(0,np.sqrt(0.5/L),(L*K,1)) # imaginary part of channel
            h = np.concatenate((h_real,h_imag),0)             # channel in real domain
            H[n,:] = h.T
            Noise = np.random.normal(0,np.sqrt(0.5*N0),(2*Tt,1)) # noise in real domain
            r = np.matmul(X,h)+Noise
            y = (r>=0.0)+0 # classes of received signals: 0 and 1

            clf.fit(X, np.squeeze(y))
            Hhat[n,:] = (np.sqrt(K)/np.linalg.norm(clf.coef_))*clf.coef_
            #Hhat[n,:] = clf.coef_
        chanEstError = (np.linalg.norm(H-Hhat)**2)/(N*K)
        NMSE[jj] = NMSE[jj] + chanEstError
    NMSE[jj] = 10*np.log10(NMSE[jj]/max_run_sim)

# Plot the NMSE               
plt.plot(snr_dB,NMSE,"-o")
plt.xlabel('SNR - dB')
plt.ylabel('NMSE - dB')
plt.grid(b=1,which='major',axis='both')
delta = 0.5
plt.axis([min_snr-delta, max_snr+delta, -19, np.max(NMSE)+delta])
print(NMSE)

filename = 'SVM_K'+str(K)+'_N'+str(N)+'_L'+str(L)+'_Np'+str(Tt)+'.txt'
np.savetxt(filename, NMSE, fmt='%.15f', delimiter=', ') # save channel list to a text file
    
    