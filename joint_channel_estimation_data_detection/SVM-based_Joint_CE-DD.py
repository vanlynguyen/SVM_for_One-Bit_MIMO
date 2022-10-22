# Written by Van Ly Nguyen, WiiLab, San Diego State University (SDSU)
# Contact: https://sites.google.com/view/vanlynguyen/home

from scipy.linalg import dft
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn import svm
import random

random.seed(100)

mod_scheme = 'QPSK' # modulation schemes: 'BPSK','QPSK','16QAM','64QAM'
if mod_scheme=='BPSK':
    Alphabet = np.array([-1.0, 1.0])
elif mod_scheme=='QPSK':
    Alphabet = np.array([-1.0-1.0j, -1.0+1.0j, 1.0-1.0j, 1.0+1.0j])
elif mod_scheme=='8PSK':
    Alphabet = np.array([np.exp(0.0), np.exp(1.0j*np.pi/4.0), np.exp(3.0j*np.pi/4.0),\
                         np.exp(1.0j*np.pi/2.0), np.exp(1.0j*7.0*np.pi/4.0), \
                         np.exp(1.0j*3.0*np.pi/2.0), np.exp(1.0j*np.pi), \
                         np.exp(1.0j*5.0*np.pi/4.0)])
elif mod_scheme=='16QAM':
    Alphabet = np.array([-3.0-3.0j, -3.0-1.0j,  -3.0+3.0j,  -3.0+1.0j, \
                         -1.0-3.0j, -1.0-1.0j,  -1.0+3.0j,  -1.0+1.0j, \
                         +3.0-3.0j, +3.0-1.0j,  +3.0+3.0j,  +3.0+1.0j, \
                         +1.0-3.0j, +1.0-1.0j,  +1.0+3.0j,  +1.0+1.0j,])

Alphabet = Alphabet/np.sqrt(np.mean(np.abs(Alphabet)**2)); # normalize symbol energy
card = Alphabet.shape[0] # constellation size
bps = np.log2(card)      # number of bits per symbol
power = np.ones((card,1))*(2**np.arange(np.log2(card)))
# bit sequence of the constellation
bits = np.floor((np.array(range(card)).reshape(card,1)%(2*power))/power).T

K = 4  # number of users, each user is equipped with only one antenna
N = 32 # number of antennas at the base station

Tb = 500      # length of the block-fading interval
Tt = 20       # length of the training sequence
Td = Tb - Tt  # length of the data sequence

min_snr  = 5.0   # minimum simulated SNR
max_snr  = 5.0   # maximum simulated SNR
snr_step = 5.0   # SNR step
snr_dB = np.arange(min_snr, max_snr+snr_step, snr_step) # snr in dB
snr = 10**(snr_dB/10.0)  # snr in linear scale

if mod_scheme=='QPSK':
    c = 1.5
    max_gamma = 3.0
elif mod_scheme=='16QAM':
    c = 1.3
    max_gamma = 1.5
ln2 = np.log(2)

NMSE_SVM = np.zeros(snr.shape,float)
NMSE_SVMjoint = np.zeros(snr.shape,float)

BER_SVM = np.zeros(snr.shape,float)
BER_SVMjoint = np.zeros(snr.shape,float)

DFT_matrix = dft(Tt)          # DFT matrix of size Tt
Theta = DFT_matrix[:,1:K+1].T # training sequence contains K columns from 
                              # the 2nd to the (K+1)th columns of DFT_matrix
# Convert Theta to the real domain, Xt is the training sequence in the real domain
Xt = np.concatenate((np.concatenate((np.real(Theta), np.imag(Theta)),1), \
                    np.concatenate((-np.imag(Theta),np.real(Theta)),1)),0)

clf = svm.LinearSVC(C = 1, fit_intercept=0, max_iter=30) # initialize a linear SVM

numChan = 2000; # number of channel realizations for the channel list
Hlist = np.random.normal(0,np.sqrt(0.5),(numChan*N,2*K)) # channel list
#filename = 'Channel_List_'+str(N)+'N_'+str(K)+'K_'+str(numChan)+'rlz.txt'
#np.savetxt(filename, Hlist, fmt='%.15f', delimiter=', ') # save channel list to a text file

Hhat_SVM = np.zeros((snr.shape[0],numChan*N,2*K),float)  # estimated channel list by SVM
Hhat_SVMjoint = np.zeros((snr.shape[0],numChan*N,2*K),float) # estimated channel list by SVM-based joint CE-DD

Hhat = np.zeros((N,2*K),float) # initialize estimated channel matrix

## ------------------------- data detection function --------------------------
def data_detection(jdx):
    Hd_hat = np.concatenate((np.concatenate(( Hhat_real.T, Hhat_imag.T),1), \
                             np.concatenate((-Hhat_imag.T, Hhat_real.T),1)),0)
    tx_idx_hat1 = np.zeros((K,Td),int)
    X_tilde_complex = np.zeros((K,Td),complex)
    for t in range(Td):
        clf.fit(Hd_hat.T, Yd[t,:])
        x_tilde_real = (np.sqrt(K)/np.linalg.norm(clf.coef_))*clf.coef_
        x_tilde_complex = x_tilde_real[0,0:K] + 1j*x_tilde_real[0,K:2*K]
        X_tilde_complex[:,t] = x_tilde_complex
        # symbol-by-symbol detection
        for k in range(K):
            min_dis = np.abs(x_tilde_complex[k]-Alphabet[0])
            for l in range(1,card):
                curr_dis = np.abs(x_tilde_complex[k]-Alphabet[l])
                if curr_dis < min_dis:
                    min_dis = curr_dis
                    tx_idx_hat1[k,t] = l

    tx_idx_hat2 = tx_idx_hat1.copy()
    Xd_complex_hat = Alphabet[tx_idx_hat2]
    X_size = 0;
    for t in range(Td):
        Xlist = []
        flag = False
        for k in range(K):
            denom = np.abs(X_tilde_complex[k,t]-Xd_complex_hat[k,t])
            currList = []
            for l in range(card):
                if Xd_complex_hat[k,t]==Alphabet[l]:
                    currList.append(Alphabet[l])
                elif np.abs(X_tilde_complex[k,t]-Alphabet[l])/denom < np.min([snr[jdx]/10+c,max_gamma]):
                    currList.append(Alphabet[l])
                    flag = True
            Xlist.append(currList)
        if flag == False:
            X_size = X_size + 1
        if flag == True:
            x_star = []
            min_wH_dis = np.inf
            for x_candidate in itertools.product(*Xlist):
                X_size = X_size + 1
                x_complex = np.asarray(x_candidate)[:,None]
                x = np.concatenate((np.real(x_complex),np.imag(x_complex)),0)
                z = np.matmul(Hd_hat.T,x)
                ck = (z>=0.0)+0
                w = snr2a*z**2 + sqrt2snrb*np.abs(z) + ln2
                w_tilde = -np.log(1.0-np.exp(-w))
                tempp = Yd[t,:]
                norm0 = (ck!=tempp[:,None])+0.0
                curr_wH_dis = np.matmul(w.T,norm0) + np.matmul(w_tilde.T,1-norm0)
                if curr_wH_dis < min_wH_dis:
                    min_wH_dis = curr_wH_dis
                    x_star = x_complex
            for k in range(K):
                for l in range(card):
                    if x_star[k] == Alphabet[l]:
                        tx_idx_hat2[k,t] = l
    X_size_mean = X_size/Td
    return tx_idx_hat1, tx_idx_hat2, X_size_mean

## ========================== SIMULATION START HERE ===========================
for jj in range(snr.shape[0]):
    print(snr_dB[jj])
    N0 = 1/snr[jj]
    snr2a = snr[jj]*2*0.374
    sqrt2snrb = np.sqrt(2*snr[jj])*0.777
    
    for ii in range(np.int(numChan)):
        if np.mod(ii,100)==0:
            print(ii)
        H_real = Hlist[ii*N:(ii+1)*N,0:K]   # real part of channel
        H_imag = Hlist[ii*N:(ii+1)*N,K:2*K] # imaginary part of channel

        H = np.concatenate((H_real,H_imag),1) # Channel in real domain
        Noise = np.random.normal(0,np.sqrt(0.5*N0),(N,2*Tt)) # noise in real domain
        Rt = np.matmul(H,Xt) + Noise # received training signal
        Yt = (Rt>=0.0)+0     # classes of received signals: 0 and 1
        
        # SVM-based channel estimation using training sequence
        for n in range(N):
            clf.fit(Xt.T, Yt[n,:])
            Hhat[n,:] = (np.sqrt(K)/np.linalg.norm(clf.coef_))*clf.coef_
        chanEstError = (np.linalg.norm(H-Hhat)**2)/(N*K)
        Hhat_SVM[jj,ii*N:(ii+1)*N,:] = Hhat.copy()
        NMSE_SVM[jj] = NMSE_SVM[jj] + chanEstError

        Hhat_real = Hhat[:,0:K].copy()
        Hhat_imag = Hhat[:,K:2*K].copy()

        # data transmission phase
        tx_idx = np.random.randint(0,card,(K,Td)) # indices of data symbols
        tx_bits = bits[:,tx_idx] # transmitted bits
        Xd_complex = Alphabet[tx_idx].T # transmitted signal in complex domain
        Xd = np.concatenate((np.real(Xd_complex),np.imag(Xd_complex)),1) # transmitted singal in real domain
        Noise = np.random.normal(0,np.sqrt(0.5*N0),(2*N,Td)).T # noise in real domain
        Hd = np.concatenate((np.concatenate(( H_real.T, H_imag.T),1), \
                             np.concatenate((-H_imag.T, H_real.T),1)),0)
        Rd = np.matmul(Xd,Hd) + Noise# received data signal
        Yd = (Rd>=0.0)+0 # classes of received data signals: 0 and 1

        tx_idx_hat1, tx_idx_hat2, X_size_mean = data_detection(jj)
        
        tx_bits_hat2 = bits[:,tx_idx_hat2] # detected bits
        BER_SVM[jj] = BER_SVM[jj] + np.sum((tx_bits_hat2!=tx_bits)+0) # error count

        Xhat = Alphabet[tx_idx_hat2]
        Xhat = np.concatenate((np.concatenate((np.real(Xhat), np.imag(Xhat)),1), \
                               np.concatenate((-np.imag(Xhat),np.real(Xhat)),1)),0)
        X = np.concatenate((Xt,Xhat),1)
        Y = np.concatenate((Yt,np.concatenate((Yd[:,0:N].T,Yd[:,N:2*N].T),1)),1)
        for n in range(N):
            clf.fit(X.T, Y[n,:])
            Hhat[n,:] = (np.sqrt(K)/np.linalg.norm(clf.coef_))*clf.coef_
        chanEstError = (np.linalg.norm(H-Hhat)**2)/(N*K)
        Hhat_SVMjoint[jj,ii*N:(ii+1)*N,:] = Hhat.copy()
        NMSE_SVMjoint[jj] = NMSE_SVMjoint[jj] + chanEstError

        Hhat_real = Hhat[:,0:K].copy()
        Hhat_imag = Hhat[:,K:2*K].copy()

        tx_idx_hat_stage1, tx_idx_hat_stage2, X_size_mean = data_detection(jj)
        tx_bits_hat2 = bits[:,tx_idx_hat_stage2] # detected bits
        BER_SVMjoint[jj] = BER_SVMjoint[jj] + np.sum((tx_bits_hat2!=tx_bits)+0) # error count

    NMSE_SVM[jj] = 10*np.log10(NMSE_SVM[jj]/numChan)
    NMSE_SVMjoint[jj] = 10*np.log10(NMSE_SVMjoint[jj]/numChan)

    BER_SVM[jj] = BER_SVM[jj]/(numChan*K*Td*np.log2(card)) # BER
    BER_SVMjoint[jj]  = BER_SVMjoint[jj]/(numChan*K*Td*np.log2(card)) # BER


# Plot the NMSE
plt.figure()
plt.semilogy(snr_dB,BER_SVM,"-*")
plt.semilogy(snr_dB,BER_SVMjoint,"-o")
plt.xlabel('SNR - dB')
plt.ylabel('BER - dB')
plt.grid(b=1,which='major',axis='both')
plt.axis([-20, 30, 1e-7, 1])

plt.figure()
plt.plot(snr_dB,NMSE_SVM,"-o")
plt.plot(snr_dB,NMSE_SVMjoint,"-*")
plt.xlabel('SNR - dB')
plt.ylabel('NMMSE - dB')
plt.grid(b=1,which='major',axis='both')