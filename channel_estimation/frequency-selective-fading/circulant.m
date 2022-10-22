function X = circulant(x)
[N,M] = size(x);
if N>1 && M == 1
    Idx = zeros(N,N);
    Idx(:,1) = 1:N;
    for n = 2:N
        Idx(:,n) = [Idx(N,n-1); Idx(1:N-1,n-1)];
    end

    X = zeros(N,N);
    for n = 1:N
        X(:,n) = x(Idx(:,n),1);
    end
else
    error('input vector must be a column vector');
end
end