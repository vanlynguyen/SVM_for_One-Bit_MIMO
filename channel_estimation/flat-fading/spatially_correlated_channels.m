%% Written by Van Ly Nguyen, WiiLab, San Diego State University (SDSU)
% Contact: https://sites.google.com/view/vanlynguyen/home

clear; clc;

K = 4; % number of transmit antennas
N = 32; % number of receive antennas

Cov_mat = readmatrix(strcat('Covariance_Data\Cov_mat_',num2str(N),'N.txt'));
Lower_mat = readmatrix(strcat('Covariance_Data\Lower_mat_',num2str(N),'N.txt'));
Lower_mat = Lower_mat(:,1:N) + 1i*Lower_mat(:,N+1:2*N);
Cov_mat = Cov_mat(:,1:N) + 1i*Cov_mat(:,N+1:2*N);

M = size(Cov_mat,1)/N;

snr_dB = -15:5:30; % snr in dB
snr = 10.^(snr_dB/10); % snr in scalar scale

NMSE_SVM = zeros(1,length(snr));

num_chan = 100;

tau = 20;
DFT_matrix = dftmtx(tau);
Theta = DFT_matrix(:,2:K+1);
X = transpose(Theta);
X_R = [real(X), imag(X); ...
       -imag(X), real(X)];
for chan_idx = 1:num_chan
    if mod(chan_idx,1)==0
        disp(chan_idx);
    end
    
    H = sqrt(0.5)*(randn(N,K)+1i*randn(N,K)); % channel
    mk_list = randi(M,1,K);
    Cinv = zeros(N*K);
    for k = 1:K
        mk = mk_list(k);
        H(:,k) = Lower_mat((mk-1)*N+1:mk*N,:)*H(:,k);
        Cinv((k-1)*N+1:k*N,(k-1)*N+1:k*N) = inv(Cov_mat((mk-1)*N+1:mk*N,:));
    end
    
    HX = H*X;
    Noise = sqrt(0.5)*(randn(N,tau)+ 1i*randn(N,tau));
    
    for jj = 1:length(snr)
        N0 = 1/snr(1,jj);
        
        R = HX + sqrt(N0)*Noise;

        Y_real = 2*(real(R)>=0)-1;
        Y_imag = 2*(imag(R)>=0)-1;
        
        cvx_clear
        cvx_begin quiet
            variable HhatSVM(N,K) complex
            variable xiRe(N,tau) nonnegative
            variable xiIm(N,tau) nonnegative
            minimize (0.5*(quad_form(HhatSVM(:,1), Cinv(1:N,1:N)) + ...
                quad_form(HhatSVM(:,2), Cinv(N+1:2*N,N+1:2*N)) + ...
                quad_form(HhatSVM(:,3), Cinv(2*N+1:3*N,2*N+1:3*N)) + ...
                quad_form(HhatSVM(:,4), Cinv(3*N+1:4*N,3*N+1:4*N))) + ...
                sum(sum(xiRe.^2+xiIm.^2)))
            subject to 
                HhatX = HhatSVM*X;
                Y_real.*real(HhatX) >= 1 - xiRe;
                Y_imag.*imag(HhatX) >= 1 - xiIm;
        cvx_end

        Hhat = sqrt(N*K)/norm(HhatSVM,'fro')*HhatSVM;
        
        NMSE_SVM(jj) = NMSE_SVM(jj) + (norm(Hhat-H,'fro')^2)/(K*N);
    end
    
end

NMSE_SVM = 10*log10(NMSE_SVM/num_chan);

figure;
plot(snr_dB,NMSE_SVM,'b-o');
xlabel('SNR in dB');
ylabel('Normalized MSE in dB');
grid on;
