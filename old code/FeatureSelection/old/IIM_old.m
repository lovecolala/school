clc; clear; close all;
%% Read Data
filename = 'IBM.csv';
Close = 4;
StockData = csvread(filename, 1, Close);
StockData = StockData(:, 1);
Delta = StockData(2:end, :) - StockData(1:end-1, :);
%% Analysis Data
% pd_n = fitdist(Delta, 'normal');
% mu = pd_n.mu;
% sigma = pd_n.sigma;
% x = linspace(mu-5*sigma, mu+5*sigma, 500);
% dx = x(2)-x(1);
% pd_k = fitdist(Delta, 'kernel');
% y = pdf(pd_k, x);
% plot(x, y, 'linewidth', 1.5);
% grid on;
% xlabel('Event'); ylabel('Probability Density');
% title('Probability Density Distribution');
%% Data pair
nData = length(Delta);
nDim = 31;
feature_table = zeros(nData-nDim+1, nDim);

feature.kernel = struct();
feature.event = zeros(1, 500);
feature.dx = 0;
feature.phi = 1;

feature = repmat(feature, 1, nDim);

for i = 1:nDim
        t = nData - nDim + i ;
        data = Delta(i:t);
        feature_table(:, i) = data;        
        
        feature(i).kernel = fitdist(data, 'kernel');
        mu = feature(i).kernel.mean;
        sigma = feature(i).kernel.std;
        
        feature(i).event = linspace( mu-5*sigma, mu+5*sigma, 500 );
        feature(i).dx = feature(i).event(2) - feature(i).event(1);
        feature(i).pd = pdf(feature(i).kernel, feature(i).event);
        feature(i).phi = max(max(feature(i).pd) + realmin, 1);
end

Entropy=@(pd, F) sum(pd .* log10( F.phi./pd )) .* F.dx;
%% Influence Information
D = zeros(nDim);
for i = 1:nDim
        
        % Feature為正及為負的真實資料
        f1.minus.data = feature_table( feature_table(:, i) < 0 );
        f1.plus.data = feature_table( feature_table(:, i) >= 0 );
        
        % 真實資料所對應的pd值 (data)
        f1.minus.pd = pdf( feature(i).kernel, f1.minus.data );
        f1.plus.pd = pdf( feature(i).kernel, f1.plus.data );
        
        % 正負5個sigma中，event為負所對應的pd值 (all)
        f1.minus.data_pd =  feature(i).pd( feature(i).event < 0 );
        f1.plus.data_pd = feature(i).pd( feature(i).event >= 0 );
        
        % 做積分→得到發生機率
        int_minus = sum( f1.minus.data_pd .* feature(i).dx );
        int_plus = sum( f1.plus.data_pd .* feature(i).dx );
        
        for j = 1:nDim
                
                % 第二個不需分正負，直接拿真實資料代入pdf中得到pd值
                pd2 = pdf( feature(j).kernel, feature_table(:, j) );
                pd2_all = pdf( feature(j).kernel, feature(j).event );
                
                % 分別計算 I( f1+ , f2 ) 及 I( f1- , f2 ) 互資訊
                I_xy_minus = InfoMutual(f1.minus.data_pd, feature(i), pd2_all, feature(j), Entropy);
                I_xy_plus = InfoMutual(f1.plus.data_pd, feature(i), pd2_all, feature(j), Entropy);
                
                D(i, j) = I_xy_minus * int_minus + I_xy_plus * int_plus;                
        end
end