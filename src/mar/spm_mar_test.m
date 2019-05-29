% Generate MAR(2) and fit MAR model

disp('Generating data from known MAR(2) model');
N_CHANNELS = 2;
AR_ORDER   = 2;
AR_ORDER_MAX_TEST = 5;
T=1000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate random dynamical system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = zeros(N_CHANNELS, 1);          % Constant term
A1 = randn(2, 2);
A2 = randn(2, 2);

% Normalize spectral radius to ensure system is stable
A1 = A1 / (max(abs(eig(A1))) * AR_ORDER * 1.1);   % Mat - lag 1
A2 = A2 / (max(abs(eig(A2))) * AR_ORDER * 1.1);   % Mat - lag 2
% disp(max(abs(eig(A1))))
% disp(max(abs(eig(A2))))
A = [ A1 A2 ];

% Noise-Covariance
C = full(sprandsym(N_CHANNELS, 1, 0.5, 1)) / 2;       
disp(eig(C))
lambda_true=inv(C);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Generate observations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = spm_mar_gen (w, A, C, T);

logev=[];
for m=1:AR_ORDER_MAX_TEST,
    disp(sprintf('Fitting MAR model with %d components',m));
    mar=spm_mar(x,m);
    logev=[logev; mar.fm];
end
logev=logev-min(logev);

disp(logev)

figure
subplot(2,1,1);
plot(x);
title('Bivariate time series from MAR(2) model');
subplot(2,1,2);
bar(logev);
xlabel('Number of time lags');
ylabel('Log Evidence');