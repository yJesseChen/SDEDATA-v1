close all
clearvars

% Simulation options
opt.format = 'epsc';
opt.my_plot = @plot;
opt.ODE = odeset('RelTol', 1e-6, 'AbsTol', 1e-4);
opt.ODE_met = @ode15s;
opt.h = true;

nam = 'CIRC73s'; 
% nam = 'KOLO';
% nam = 'p53';
[pr, pa] = examples(nam);


% % training data
% T = 0.1;
% Nt = 1;
% N_data = 200000;
% xICinfo = [3500,600,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ; 5000,1400,700,1200,5500,1800,600,2700,1000,100,3200,300,3000,400,300,600];

% t = linspace(0,T,Nt+1);

% dim = size(xICinfo,2);
% xIC = zeros(N_data,dim);
% for i=1:dim
%     xIC(:,i) = randi([xICinfo(1,i), xICinfo(2,i)],1,N_data);
% end
% data = GenDatawithGillespieSSA(pr, pa, T, xIC, t, opt);
% data = int32(data);
% save("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_train.mat","data");

% % testing data
% T = 100;
% Nt = 1000;
% N_data = 1000;

% t = linspace(0,T,Nt+1);
% % ddd = pr.S0;
% ddd = [4250,1000,350,600,2750,900,300,1350,500,50,1600,150,1500,200,150,300];
% xIC = repmat(ddd,N_data,1);
% data = GenDatawithGillespieSSA(pr, pa, T, xIC, t, opt);
% data = int32(data);
% save("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_test.mat","data");

% % condition data
% T = 0.1;
% Nt = 1;
% N_data = 10000;
% xICinfo = [pr.S0; 4250,1000,350,600,2750,900,300,1350,500,50,1600,150,1500,200,150,300;4000,600,175,300,1375,450,150,675,250,25,800,75,750,100,75,150;5000,1400,500,1000,5000,1000,400,1700,600,50,1800,200,1500,300,200,400;];

% t = linspace(0,T,Nt+1);
% dim = size(xICinfo,2);
% xinN = size(xICinfo,1);

% for i=1:xinN
%    xIC = repmat(xICinfo(i,:),N_data,1);
%    data = GenDatawithGillespieSSA(pr, pa, T, xIC, t, opt);
%    data = int32(data(:,2,:));
%    data = reshape(data,dim,N_data).';
%    assignin('base',join(["m",string(i-1),'_i'],""),int32(xICinfo(i,:)));
%    save("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_cond.mat",join(["m",string(i-1),'_i'],""),"-append");

%    assignin('base',join(["m",string(i-1),'_d'],""),data);
%    save("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_cond.mat",join(["m",string(i-1),'_d'],""),"-append");

%    figure
%    tiledlayout(4,4)
%    histogram(data(:,1))
%    for j=2:16
%        nexttile
%        histogram(data(:,j))
%    end
% end

% size = [sqrt(xinN),sqrt(xinN)];
% save("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_cond.mat","size","-append");

% K = 4
% for i in range(K):
%     a[str(i)+'_i'] = a['m'+str(i)+'_i']
%     del a['m'+str(i)+'_i']
%     a[str(i)+'_d'] = a['m'+str(i)+'_d']
%     del a['m'+str(i)+'_d']
% for i in range(K):
%     a[str(i)+'_i'] = np.squeeze(a[str(i)+'_i'])
%     a[str(i)+'_d'] = np.squeeze(a[str(i)+'_d'])
% sio.savemat("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_cond.mat",a)

% Ori data
T = 100;
N_data = 10;

vague = 10;
save("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_Ori.mat","vague");

% ddd = pr.S0;
ddd = [4250,1000,350,600,2750,900,300,1350,500,50,1600,150,1500,200,150,300];
for i=1:N_data
	[react_time,react_numb] = SSA_path_rs(pr, pa, T, ddd, opt);
	react_numb = int32(react_numb);
	assignin('base',join(["t",'_',string(i-1)],""),react_time);
	save("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_Ori.mat",join(["t",'_',string(i-1)],""),"-append");

	assignin('base',join(["d",'_',string(i-1)],""),react_numb);
	save("/Users/jesse/Dropbox/DataProd/Ex41SSACIRC73s_Ori.mat",join(["d",'_',string(i-1)],""),"-append");
end



% // T = 200; % Final time
% // M = 1; % Number of trajectories

% // figure;
% // hold on;

% // for m = 1:M
% //     tic
% //     [ts_ex, ss_ex] = SSA_path_rs(pr, pa, T, pr.S0, opt);
% //     toc
    
% //     plot(ts_ex, ss_ex);
% // end

% // title(['Stochastic trajectories of ', nam, ' (N = ', num2str(sum(pr.S0)), ')'], 'Interpreter', 'latex');
% // xlabel('$t$', 'Interpreter', 'latex');
% // ylabel('Species particle count', 'Interpreter', 'latex');
% // hold off;

% // saveas(gcf, ['trajectories_', nam], opt.format);

% // [ts, ss] = opt.ODE_met(@pr.ODE, [0, T], pr.S0, opt.ODE, pa.ks);

% // figure;
% // plot(ts, ss);
% // title(['Limit trajectory of ', nam], 'Interpreter', 'latex');
% // xlabel('$t$', 'Interpreter', 'latex');
% // ylabel('Sepcies density', 'Interpreter', 'latex');

% // saveas(gcf, ['ODE_', nam], opt.format);

% // rt = compute_pFIM_fastv2(pr, pa.ks, ts, ss, true, false);

% // figure;
% // bar3(rt);
% // title(['Pathwise sensitivity limit of ', nam], 'Interpreter', 'latex');
% // axis tight;
