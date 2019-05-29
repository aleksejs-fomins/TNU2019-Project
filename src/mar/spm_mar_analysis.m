%%%%%%%%%%%%%%%%%%%
% Get Data
%%%%%%%%%%%%%%%%%%%
[file,path] = uigetfile('./*.h5');
data = hdf5read([path file], 'megadata');
[N_CHANNELS, N_TIMES, N_TRIALS] = size(data);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Test models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

AR_ORDER_MAX_TEST = 1;

logev_trials = [];
amat_trials = [];

for iTrial = 1:N_TRIALS
    dataThis = data(1:12,:,iTrial).';
    logev = [];
    amat = [];
    for iOrder = 1:AR_ORDER_MAX_TEST
        disp(sprintf('trial %d : fitting MAR model with %d components', iTrial, iOrder));
        mar = spm_mar(dataThis, iOrder);
        logev=[logev mar.fm];
        amat = [amat mar.lag];
    end
    logev=logev-min(logev);
    logev_trials=[logev_trials; logev];
    amat_trials = [amat_trials; amat];
end


%% Plot Stuff
mu_logev  = mean(logev_trials);
std_logev = std(logev_trials);

figure
subplot(1,1,1);
depth_x = linspace(1,AR_ORDER_MAX_TEST,AR_ORDER_MAX_TEST);
errorbar(depth_x, mu_logev, std_logev, 'o','LineWidth',2);
xlabel('Number of time lags');
ylabel('Log Evidence');
set(findall(gcf,'-property','FontSize'),'FontSize',18)
axis([0 AR_ORDER_MAX_TEST+1 -max(std_logev) max(mu_logev)+max(std_logev)])

%% Save results
[file,path] = uiputfile('./*.mat');
save([path file], 'amat_trials')