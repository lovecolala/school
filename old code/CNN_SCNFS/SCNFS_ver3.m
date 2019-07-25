clc; clear; close all;
%% Option
opts.nAnt = 10;
opts.new_nAnt = 30;
opts.eva_rate = 0.9;
opts.learning_rate = 0.1;

nTarget = 4;
Ra = 0.33;
% Ra = 0.1;
upper = 4;
iteration = 100;

%% Read Data
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
rmse =@(error) ( (error*error') / length(error) ) ^ 0.5;
filename = {'IBM.csv', 'IBM'
        'AAPL.csv', 'Apple'
        'DVMT.csv', 'DELL'
        'T.csv', 'AT&T'};
Close = 4;
StockData = [];
for i = 1:nTarget
        t = csvread(filename{i, 1}, 1, Close);
        t = t(:, 1);
        StockData = [StockData, t];
end

%% Feature Selection
nFeature = 30;
[feature_table, FP] = FeatureSelection(StockData, nFeature);
FP = FP( 1:min(length(FP), upper) );
FP = sort(FP);

%% Target
nOutput = ceil(nTarget/2);
nInput = size(feature_table, 1);

s = StockData(2:end, :);
index = repmat(0:nFeature, nInput, 1) + (1:nInput)';
index_feature = index(:, 1:end-1);
index_target = index(:, end);
data_table = [];
target_table = [];
for i = 1:nTarget
        t = s(:, i);
        data_table = [data_table, t(index_feature)];
        target_table = [target_table, t(index_target)];
end
data_table = [data_table, target_table];

% 220-by-2
target = zeros(nInput, nOutput);
for i = 1:nTarget
        t = data_table(:, end-nTarget+i);
        if mod(i, 2) == 0
                t = t .* sqrt(-1);
        end
        target(:, ceil(i/2)) = target(:, ceil(i/2)) + t;
end

%% input
nDim = length(FP);
% nDim -by- nInput
input = data_table(:, FP)';

%% Subtractive Clustering
% Center and Sigma
h = Subclustering(input, Ra);

%% Construct Matrix
t = []; comma = [];
for i = 1:nDim
        t = [ t, comma, '1:length(h{', int2str(i), '}.center)' ];
        comma = ', ';
end
eval(['C = allcomb(', t, ');']);

%% Block Selection
C = BlockSelection(C, input, h, nTarget);

%% contruct index and parameter
[Rule_index, para] = ConIndex(C, 0, h);
ant(1).position = para;
for i = 1:opts.nAnt
        ant(i).position = para .* abs(randn(size(para)));
end

%% Training
all_nAnt = opts.nAnt + opts.new_nAnt;
nRule = size(C, 1);
theta = zeros(nRule*(nDim+1), nOutput);
LearnCurve = zeros(1, iteration);
output = zeros(size(target));

weight = gauF(1:opts.nAnt, 1, opts.learning_rate*opts.nAnt) ./ (opts.learning_rate*opts.nAnt*sqrt(2*pi));
opts.prob = weight ./ sum(weight);
beta = zeros(nRule, nInput, nOutput);

for i = 1:iteration
        %% New ANT
        disp(['----------  iteration =', int2str(i), '  ----------']);
        ant = newant(ant, opts);
        
        for j = 1:all_nAnt
                %% Beta
                for k = 1:nRule
                        Center = Rule_index.center(k, :);
                        CenterSigma = ant(j).position([Center', (Center+1)']);
                        beta(k, :, :) = SphereCom(input, CenterSigma, nTarget);
                end
                
                %% RLSE
                A = repmat([ones(nInput, 1), input'], 1, nRule);
                for k = 1:nOutput
                        A_beta = A .* repelem( transpose(beta(:, :, k)), 1, 1+nDim );
                        theta(:, k) = RLSE(A_beta, transpose(target(:, k)), theta(:, k));
                        output(:, k) = A_beta * theta(:, k);
                end
                
                %% RMSE
                error = target - output;
                RMSE(j) = rmse(transpose(error(:)));
                
        end
        %% Sorting
        [RMSE, sorting] = sort(RMSE);
        sorting = sorting(1:opts.nAnt);
        ant = ant(sorting);
        LearnCurve(i) = RMSE(1);
        disp(['RMSE = ', num2str(RMSE(1))]);
        
end

%% Drawing
figure;
semilogy(LearnCurve, 'linewidth', 1.5);
grid on; axis tight;
title('Learning Curve');
xlabel('x'); ylabel('RMSE');

figure('Name', 'SCNFS', 'NumberTitle', 'off');
date = (1:nInput) + nFeature;
for i =1:nTarget
        t1 = factor(nTarget);
        t1 = t1(1);
        t2 = nTarget / t1;
        subplot(t2, t1, i);
        plot(StockData(:, i), 'linewidth', 1.5);
        title(filename{i, 2});
        xlabel('Date'); ylabel('Stock price');
        grid on; hold on;
        
        model_output = transpose(output(:, ceil(i/2)));
        if mod(i, 2) == 0
                model_output = imag(model_output);
        else
                model_output = real(model_output);
        end
        
        plot(date, model_output, '--', 'linewidth', 1.5);
        axis tight; hold off;
end

figure;
for i = 1:nTarget
        subplot(t2, t1, i);
        if mod(i, 2) == 0
                e = imag(error(:, ceil(i/2)));
        else
                e = real(error(:, ceil(i/2)));
        end
        plot(1:nInput, e);
        title([filename{i, 2}, ' error']);
        xlabel('date'); ylabel('error');
        grid on; axis tight;
end