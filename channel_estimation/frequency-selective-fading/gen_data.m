clear; clc;

K = 2;
N = 16;
L = 8;

factor_extra_pilots = 16;

snr_dB = -15:5:30;
snr = 10.^(snr_dB/10);


Np = K*L*factor_extra_pilots;
pilot_seq = exp(1j*rand(Np,1)*2*pi);
Xt = zeros(Np,K);
k = 1;
for n = 1:Np
    Xt(n,k) = sqrt(K)*pilot_seq(n);
    if k == K
        k = 1;
    else
        k = k + 1;
    end
end

theta = zeros(Np,K);
for k = 1:K
    theta(:,k) = ifft(Xt(:,k))*sqrt(Np);
end

Theta_L = zeros(Np,K*L);
for k = 1:K
    Theta_kL = circulant(theta(:,k));
    Theta_L(:,(k-1)*L+1:k*L) = Theta_kL(:,1:L);
end

dlmwrite('real_Theta_L.txt', real(Theta_L), 'delimiter', ',');
dlmwrite('imag_Theta_L.txt', imag(Theta_L), 'delimiter', ',');
dlmwrite('Np.txt', Np);
dlmwrite('L.txt', L);