close all
clearvars

% Simulation options
opt.format = 'epsc';
opt.my_plot = @plot;
opt.ODE = odeset('RelTol', 1e-6, 'AbsTol', 1e-4);
opt.ODE_met = @ode15s;
opt.h = true;

nam = 'Vilar2002R'; 
% nam = 'KOLO';
% nam = 'p53';
[pr, pa] = examples(nam);


% % training data
% T = 0.1;
% Nt = 1;
% N_data = 200000;
% xICinfo = [0,0,0,0,0,0,0,0,0 ; 5,5,5,5,80,2000,120,2700,2500];

% t = linspace(0,T,Nt+1);

% dim = size(xICinfo,2);
% xIC = zeros(N_data,dim);
% for i=1:dim
%     xIC(:,i) = randi([xICinfo(1,i), xICinfo(2,i)],1,N_data);
% end
% data = GenDatawithGillespieSSA(pr, pa, T, xIC, t, opt);
% data = int32(data);
% save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_train.mat","data");

% % training data extended
% T = 0.1;
% Nt = 1;
% N_data = 200000;
% % xICinfo = [-10,-10,-10,-10,-80,-2000,-120,-2700,-2500 ; 10,10,10,10,80,2000,120,2700,2500];
% xICinfo = [-2,-2,-2,-2,-40,-1000,-60,-1350,-1250 ; 2,2,2,2,80,2000,120,2700,2500];

% t = linspace(0,T,Nt+1);

% dim = size(xICinfo,2);
% xIC = zeros(N_data,dim);
% for i=1:dim
%     xIC(:,i) = randi([xICinfo(1,i), xICinfo(2,i)],1,N_data);
% end
% xIC_ = max(xIC,0);
% data = GenDatawithGillespieSSA(pr, pa, T, xIC_, t, opt);
% data = int32(data);
% % data(:,1,:) = reshape(xIC.',dim,1,N_data);
% save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_Extd_train.mat","data");

% % training data trajectory -> pair-wise
% T = 100;
% Nt = 1000;
% N_data = 2000;
% xICinfo = [1,1,0,0,0,0,0,0,0 ; 3,3,0,0,80,2000,120,2700,2500];

% t = linspace(0,T,Nt+1);

% dim = size(xICinfo,2);
% xIC = zeros(N_data,dim);
% for i=1:dim
%     xIC(:,i) = randi([xICinfo(1,i), xICinfo(2,i)],1,N_data);
% end
% data = GenDatawithGillespieSSA(pr, pa, T, xIC, t, opt);
% data = int32(data);

% % Nrandtake = 5;
% % N_pair = N_data*Nrandtake;
% % data_pair = zeros(dim,2,N_pair);
% % for i=1:N_data
% %     s_indice = randi([1, Nt],N_data);
% %     data_pair(:,1,) = data(:,s_indice,i);
% %     data_pair(:,2,) = data(:,s_indice+1,i);
% % end

% save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_manifold_train.mat","data");

% % testing data
% T = 200;
% Nt = 2000;
% N_data = 200;

% t = linspace(0,T,Nt+1);
% % ddd = pr.S0;
% ddd = [1,1,0,0,40,1000,60,1350,1250];
% xIC = repmat(ddd,N_data,1);
% data = GenDatawithGillespieSSA(pr, pa, T, xIC, t, opt);
% data = int32(data);
% save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_test.mat","data");

% % condition data
% T = 0.1;
% Nt = 1;
% N_data = 10000;
% % T=0.1
% xICinfo = [pr.S0; 1,1,0,0,40,1000,60,1350,1250;
%             0,0,1,1,40,1000,60,1350,1250;
%             1,1,0,0,20,500,30,1000,600;
%             1,    1,    0,    0,    5,    0,    0, 1834,  319;
%             1,   1,   0,   0,   2,   0,   4, 189, 262;
%             1,   1,   0,   0,   5,   0,   0, 199, 206;
%             0,    0,    1,    1,   43, 1150,   29,    0,  448;
%             0,    0,    1,    1,   52, 1385,   64,    0,  678;
%             1,   1,   0,   0,   4,   1,   2, 112, 275;
%             0,   1,   1,   0,  11,   2,   2, 115, 272;
%             1,   1,   0,   0,  23,   8,   1,  56, 331];
% % T = 0.05
% % xICinfo = [pr.S0; 1,1,0,0,40,1000,60,1350,1250;
% %             0,0,1,1,40,1000,60,1350,1250;
% %             1,1,0,0,20,500,30,1000,600;
% %             1,    1,    0,    0,    5,    0,    0, 1834,  319;
% %             1,   1,   0,   0,   2,   0,   4, 189, 262;
% %             1,   1,   0,   0,   5,   0,   0, 199, 206;
% %             0,    0,    1,    1,   43, 1150,   29,    0,  448;
% %             0,    0,    1,    1,   52, 1385,   64,    0,  678;
% %             1,   1,   0,   0,  10,   6,   1,  46, 331;
% %             0,   1,   1,   0,  13,   7,   1,  38, 339;
% %             1,   1,   0,   0,  19,  24,   2,  17, 361];
% % xICinfo = [  1,   1,   0,   0,  10,   6,   1,  46, 331];

% t = linspace(0,T,Nt+1);
% dim = size(xICinfo,2);
% xinN = size(xICinfo,1);

% emptyStruct = struct;
% save('/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_cond.mat','-struct','emptyStruct');

% for i=1:12
%    xIC = repmat(xICinfo(i,:),N_data,1);
%    data = GenDatawithGillespieSSA(pr, pa, T, xIC, t, opt);
%    data = int32(data(:,2,:));
%    data = reshape(data,dim,N_data).';
%    assignin('base',join(["m",string(i-1),'_i'],""),int32(xICinfo(i,:)));
%    save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_cond.mat",join(["m",string(i-1),'_i'],""),"-append");

%    assignin('base',join(["m",string(i-1),'_d'],""),data);
%    save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_cond.mat",join(["m",string(i-1),'_d'],""),"-append");

%    % figure
%    % tiledlayout(4,4)
%    % histogram(data(:,1))
%    % for j=2:9
%    %     nexttile
%    %     histogram(data(:,j))
%    % end
% end

% size = [sqrt(xinN),sqrt(xinN)];
% save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_cond.mat","size","-append");

% K = 12
% for i in range(K):
%     a[str(i)+'_i'] = a['m'+str(i)+'_i']
%     del a['m'+str(i)+'_i']
%     a[str(i)+'_d'] = a['m'+str(i)+'_d']
%     del a['m'+str(i)+'_d']
% for i in range(K):
%     a[str(i)+'_i'] = np.squeeze(a[str(i)+'_i'])
%     a[str(i)+'_d'] = np.squeeze(a[str(i)+'_d'])
% sio.savemat("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_cond.mat",a)



% % condition data trail
% T = 0.1;
% Nt = 1;
% N_data = 10000;
% xICinfo = [     1,   1,   0,   0,   3,   1,   7, 774, 179];

% t = linspace(0,T,Nt+1);
% dim = size(xICinfo,2);
% xinN = size(xICinfo,1);

% xIC = repmat(xICinfo,N_data,1);
% data = GenDatawithGillespieSSA(pr, pa, T, xIC, t, opt);
% data = int32(data(:,2,:));
% data = reshape(data,dim,N_data).';
% save("/Users/jesse/Desktop/a.mat","data");

% Ori data
T = 200;
N_data = 10;

vague = 10;
save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_Ori.mat","vague");

% ddd = pr.S0;
ddd = [1,1,0,0,40,1000,60,1350,1250];
for i=1:N_data
	[react_time,react_numb] = SSA_path_rs(pr, pa, T, ddd, opt);
	react_numb = int32(react_numb);
	assignin('base',join(["t",'_',string(i-1)],""),react_time);
	save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_Ori.mat",join(["t",'_',string(i-1)],""),"-append");

	assignin('base',join(["d",'_',string(i-1)],""),react_numb);
	save("/Users/jesse/Dropbox/DataProd/Ex42Vilar2002R_Ori.mat",join(["d",'_',string(i-1)],""),"-append");
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
