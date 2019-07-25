clc; clear; close all;
%% Option
opts.nAnt = 5;
opts.new_nAnt = 10;
opts.eva_rate = 0.9;
opts.learning_rate = 0.5;

nTarget = 4;
Ra = 0.25;
% Ra = 0.15;
iteration = 100;

% 是(1)否(0)要做Block Selection
opts.blockselect = 1;

%% Read Data
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;
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
if mod(length(FP), 2) == 1
        FP = FP(1:end-1);
end
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
        t = s((1:nInput)+nFeature, i);
        if mod(i, 2) == 0
                t = t .* sqrt(-1);
        end
        target(:, ceil(i/2)) = target(:, ceil(i/2)) + t;
end

%% Data processing
nFP = length(FP);
cnn_input = data_table(:, FP);
if mod(nFP, 4) == 0
        cnn_input = reshape(cnn_input', 4, nFP/4, []);
else
        cnn_input = reshape(cnn_input', 2, nFP/2, []);
end

%% Set the parameter
weight = gauF(1:opts.nAnt, 1, opts.learning_rate*opts.nAnt) ./ (opts.learning_rate*opts.nAnt*sqrt(2*pi));
opts.prob = weight ./ sum(weight);

%% CNN
if nTarget == 1
        cnn.layers={
                struct('type', 'i')
                struct('type', 'c', 'nFilter', 1, 'filtersize', [2, 1])
                struct('type', 'c', 'nFilter', 1, 'filtersize', [1, 2])
                struct('type', 'c', 'nFilter', 1, 'filtersize', [1, 2])
                };
elseif nTarget == 2
        cnn.layers={
                struct('type', 'i')
                struct('type', 'c', 'nFilter', 1, 'filtersize', [2, 1])
                struct('type', 'c', 'nFilter', 1, 'filtersize', [2, 1])
                struct('type', 'c', 'nFilter', 1, 'filtersize', [2, 1])
                };
elseif nTarget == 3
        cnn.layers={
                struct('type', 'i')
                struct('type', 'c', 'nFilter', 1, 'filtersize', [2, 3])
                struct('type', 'c', 'nFilter', 1, 'filtersize', [1, 5])
                struct('type', 'c', 'nFilter', 1, 'filtersize', [1, 5])
                };
elseif nTarget == 4
        cnn.layers={
                struct('type', 'i')
                struct('type', 'c', 'nFilter', 1, 'filtersize', [2, 3])
                struct('type', 'c', 'nFilter', 1, 'filtersize', [3, 2])
                struct('type', 'c', 'nFilter', 1, 'filtersize', [1, 5])
                };
end
cnn.layers{1}.output{1} = cnn_input;

[swarm, cnn] = cnn_initial(cnn, opts);
cnn = cnn_fp( cnn, swarm, ones(1, numel(swarm)) );

%% reshape CNN
% 到SCNFS的維度數，也就是CNN的output數
CnnOutput = [];
for i = 1:numel(cnn.layers{end}.output)
        sizeA = size(cnn.layers{end}.output{i});
        t = reshape(cnn.layers{end}.output{i}, sizeA(1)*sizeA(2), sizeA(3));
        CnnOutput = [CnnOutput; t];
end
nDim = size(CnnOutput, 1);

%% Subtractive Clustering
% Center and Sigma
h = Subclustering(CnnOutput, Ra);

%% Construct Matrix
t = []; comma = [];
for i = 1:nDim
        t = [ t, comma, '1:length(h{', int2str(i), '}.center)'];
        comma = ', ';
end
eval(['C = allcomb(', t, ');']);

%% Block Selection
if opts.blockselect
        C = BlockSelection(C, CnnOutput, h, nTarget);
end

%% construct index and parameter
[Rule_index, para] = ConIndex(C, 0, h);
swarm(end+1).ant(1).position = para;
for i = 2:opts.nAnt
        swarm(end).ant(i).position = para .* abs(randn(size(para)));
end

%% Training
all_nAnt = opts.nAnt + opts.new_nAnt;
nRule = size(C, 1);
theta = zeros(nRule.*(nDim+1), nOutput);
LearnCurve = zeros(1, iteration*numel(swarm));
output = zeros(size(target));
beta = zeros(nRule, nInput, nOutput);

for i = 1:iteration
        disp(['----------   iteration = ', int2str(i), '  ----------']);
        swarm = antrenew(swarm, opts);
        for j = 1:numel(swarm)
                for k = 1:all_nAnt
                        %% CNN
                        ant_index = ones(1, numel(swarm));
                        ant_index(j) = k;
                        cnn = cnn_fp( cnn, swarm, ant_index );
                        CnnOutput = [];
                        for q = 1:numel(cnn.layers{end}.output)
                                sizeA = size(cnn.layers{end}.output{q});
                                t = reshape(cnn.layers{end}.output{q}, sizeA(1)*sizeA(2), sizeA(3));
                                CnnOutput = [CnnOutput; t];
                        end
                        
                        %% Beta
                        % beta = [nRule, nInput, nOutput]
                        for q = 1:nRule
                                Center = Rule_index.center(q, :);
                                CenterSigma = swarm(end).ant(ant_index(end)).position([Center', (Center+1)']);
                                beta(q, :, :) = SphereCom(CnnOutput, CenterSigma, nTarget);
                        end
                        
                        %% RLSE
                        A = repmat([ones(nInput, 1), CnnOutput'], 1, nRule);
                        for q = 1:nOutput
                                A_beta = A .* repelem( transpose(beta(:, :, q)), 1, 1+nDim );
                                theta(:, q) = RLSE(A_beta, transpose(target(:, q)), theta(:, q));
                                output(:, q) = A_beta * theta(:, q);
                        end
                        
                        %% RMSE
                        error = target - output;
                        RMSE(k) = rmse(error(:));
                        swarm(j).ant(k).RMSE = RMSE(k);
                end
                %% Sorting
                [RMSE, sorting] = sort(RMSE);
                sorting = sorting(1:opts.nAnt);
                swarm(j).ant = swarm(j).ant(sorting);
                LearnCurve((i-1)*numel(swarm)+j) = RMSE(1);
                disp(['RMSE = ', num2str(RMSE(1))]);
                
        end
end

%% Drawing
% Learning Curve
figure;
semilogy(LearnCurve, 'linewidth', 1.5);
grid on; axis tight;
title('Learning Curve');
xlabel('x'); ylabel('RMSE');

% Stock
figure('Name', 'CNN+SCNFS', 'NumberTitle', 'off');
date = (1:nInput) + nFeature;
for i = 1:nTarget
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