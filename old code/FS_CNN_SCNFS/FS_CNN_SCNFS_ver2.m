clc; 
clear; close all;
%% Add path
addpath(genpath('Stock data'));
addpath(genpath('function'));
addpath(genpath('record'));
current_path = pwd;

%% Option
Ra = 0.3;
iteration = 30;
nFeature = 30;
nTarget = 4;

opts.nAnt = 5;
opts.new_nAnt = 10;
opts.eva_rate = 0.9;
opts.learning_rate = 0.1;

opts.blockselection = 1;
opts.beta_type = 'Sphere';
% Sphere/Standard
opts.input_type = 'delta';
% actural_price/delta

nFP = 361;
cnn_input_size = 19;

%% Read Data
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;

filename = {'2330.TW.csv', 'tsmc'
        '^TWII.csv', 'TAIEX'
        'GOOGL.csv', 'Google'
        '^GSPC.csv', 'S&P 500'};

rowdata = cell(1, nTarget);
for i = 1:nTarget
        rowdata{i} = csvread(filename{i, 1}, 1, 0);
end
Close = 4;

%% Intersect Data
intersect_date = rowdata{1}(:, 1);
for i = 2:nTarget
        intersect_date = intersect(intersect_date, rowdata{i}(:, 1));
end
nDate = length(intersect_date);
if strcmp(opts.input_type, 'delta')
        % 若為delta則需做差分
        nData = nDate - 1;
elseif strcmp(opts.input_type, 'actural_price')
        nData = nDate;
else
        disp('error in input_type');
end
nInput = nData - nFeature;
nTraining = ceil(nInput*0.9);

%% Transfer Data
nPrice = size(rowdata{1}, 2)-2;

index = reshape(1:nFeature*nPrice, 4, []);
index = repmat(index, 1, 1, nInput) + reshape((0:nInput-1).*nPrice, 1, 1, []);

data = zeros(nDate, nPrice, nTarget);
cnn_input = zeros(nPrice, nFeature, nTarget, nInput);
model_target = zeros(nInput, nTarget);

for i = 1:nTarget
        [~, t] = intersect(rowdata{i}(:, 1), intersect_date);
        % 1. date
        % 2. 最高/最低/開盤/收盤
        % 3. tsmc/TAIEX/Google/S&P 500
        data(:, :, i) = rowdata{i}(t, 2:end-1);
        if strcmp(opts.input_type, 'actual_price')
                temp = data(:, :, i)';                                   % 4*236
        elseif strcmp(opts.input_type, 'delta')
                temp = data(2:end, :, i)' - data(1:end-1, :, i)';                                   % 4*235
        end
        for j = 1:nInput
                % 1. 最高/最低/開盤/收盤, 
                % 2. 1:nFeature, 
                % 3. tsmc/TAIEX/Google/S&P 500, 
                % 4. 1:nInput
                cnn_input(:, :, i, j) = temp(index(:, :, j));
        end
        model_target(:, i) = temp(Close, nFeature+1:end);
end

%% Target (complex)
nOutput = ceil(nTarget/2);              % 4
nInput = size(model_target, 1);        % 205

target = zeros(nInput, nOutput);
for i = 1:nTarget
        t = model_target(:, i);
        if mod(i, 2) == 0
                t = t .* sqrt(-1);
        end
        target(:, ceil(i/2)) = target(:, ceil(i/2)) + t;
end

%% Set the parameter
weight = gauF(1:opts.nAnt, 1, opts.learning_rate*opts.nAnt) ./ (opts.learning_rate*opts.nAnt*sqrt(2*pi));
opts.prob = weight ./ sum(weight);

%% CNN structure
% input_size = 4*30*4
cnn.layers={
        struct('type', 'i')
        struct('type', 'c', 'nFilter', 1, 'filtersize', [1, 5, 1])              % 4*26*4
        struct('type', 'p', 'stride', [2, 2, 1]);                                     % 2*13*4
        struct('type', 'c', 'nFilter', 1, 'filtersize', [1, 4, 1])              % 2*10*4
        struct('type', 'p', 'stride', [2, 2, 1]);                                     % 1*5*4
        struct('type', 'c', 'nFilter', 1, 'filtersize', [1, 5, 1])              % 1*1*4
        };
cnn.layers{1}.output{1} = cnn_input;

[swarm, cnn] = cnn_initial(cnn, opts);
cnn = cnn_fp_multiswarm(cnn, swarm, ones(1, numel(swarm)));

%% reshape CNN
CnnOutput = cnn_transfer(cnn);
nDim = size(CnnOutput, 1);

%% Subtractive Clustering
h = Subclustering(CnnOutput(:, 1:nTraining), Ra);

%% Construct Matrix
t = []; comma = [];
for i = 1:nDim
        t = [ t, comma, '1:length(h{', int2str(i), '}.center)'];
        comma = ', ';
end
eval(['C = allcomb(', t, ');']);

%% Block Selection
if opts.blockselection
        C = BlockSelection(C, CnnOutput(:, 1:nTraining), h, nTarget, opts.beta_type);
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
nSwarm = numel(swarm);

LearnCurve = zeros(1, iteration*numel(swarm));

output = zeros(nTraining, nOutput);
beta = zeros(nRule, nTraining, nOutput);
theta = zeros(nRule.*(nDim+1), nOutput);
RMSE = zeros(1, all_nAnt);

A = repmat([ones(nInput, 1), CnnOutput'], 1, nRule);
A_train = A(1:nTraining, :);

tic;
for i = 1:iteration
        disp(['---------- iteration = ', int2str(i), ' ----------']);
        % 藉由原始的 "nAnt" 個位置產生出 "new_nAnt" 個新位置
        swarm = antrenew(swarm, opts);
        for j = 1:nSwarm          % 群數
                temp_theta = cell(1, all_nAnt);
                for k = 1:all_nAnt              % 一群下的螞蟻個數，也可用numel(swarm(j))
                        %% CNN
                        ant_index = ones(1, nSwarm);
                        ant_index(j) = k;
                        cnn = cnn_fp_multiswarm(cnn, swarm, ant_index);
                        CnnOutput = cnn_transfer(cnn);
                        
                        %% Beta
                        for q = 1:nRule
                                Center = Rule_index.center(q, :);
                                CenterSigma = swarm(end).ant(ant_index(end)).position([Center', (Center+1)']);
                                input = CnnOutput(:, 1:nTraining);
                                if strcmp(opts.beta_type, 'Sphere')
                                        beta(q, :, :) = SphereCom(input, CenterSigma, nTarget);
                                elseif strcmp(opts.beta_type, 'Standard')
                                        beta(q, :, :) = FirStrg(input, CenterSigma);
                                end
                        end
                        
                        %% Normalization
                        for q = 1:nOutput
                                t = beta(:, :, q);
                                t = t ./ sum(t, 1);
                                beta(:, :, q) = t;
                        end
                        
                        %% RLSE
                        for q = 1:nOutput
                                A_beta = A_train .* repelem(transpose(beta(:, :, q)), 1, 1+nDim);
                                temp_theta{k}(:, q) = RLSE(A_beta, transpose(target(:, q)), theta(:, q));
                                output(:, q) = A_beta * temp_theta{k}(:, q);
                        end
                        
                        %% RMSE
                        error = (target(1:nTraining, :) - output);
                        RMSE(k) = rmse(error(:));
                        swarm(j).ant(k).RMSE = RMSE(k);
                        
                end
                %% Sorting
                [RMSE, sorting] = sort(RMSE);
                sorting = sorting(1:opts.nAnt);
                swarm(j).ant = swarm(j).ant(sorting);
                LearnCurve((i-1)*numel(swarm)+j) = RMSE(1);
%                 theta = temp_theta{sorting(1)};
                disp(['RMSE = ', num2str(RMSE(1))]);
                
        end
end
theta = temp_theta{sorting(1)};
disp(['Wasted time = ', num2str(toc), ' sec']);

%% Testing
beta = zeros(nRule, nInput, nOutput);
output = zeros(nInput, nOutput);
ant_index = ones(1, nSwarm);
cnn = cnn_fp_multiswarm(cnn, swarm, ant_index);
CnnOutput = cnn_transfer(cnn);
% Calculate Beta
for i = 1:nRule
        Center = Rule_index.center(i, :);
        CenterSigma = swarm(end).ant(ant_index(end)).position([Center', (Center+1)']);
        if strcmp(opts.beta_type, 'Sphere')
                beta(i, :, :) = SphereCom(CnnOutput, CenterSigma, nTarget);
        elseif strcmp(opts.beta_type, 'Standard')
                beta(i, :, :) = FirStrg(CnnOutput, CenterSigma);
        end
end
% Normalization
for i = 1:nOutput
        t = beta(:, :, i);
        t = t ./ sum(t, 1);
        beta(:, :, i) = t;
end
% use theta calculate output
for i = 1:nOutput
        A_beta = A .* repelem(transpose(beta(:, :, i)), 1, 1+nDim);
        output(:, i) = A_beta * theta(:, i);
end
% error = (target - output) .* normal;
error = target - output;

%% The end of RMSE
train_error = error(1:nTraining, :);
test_error = error(nTraining+1:end, :);
result_RMSE = [rmse(train_error(:)), rmse(test_error(:)), rmse(error(:))];
disp('----------------------------------------');
disp(['Training Data RMSE = ', num2str(result_RMSE(1))]);
disp(['Testing Data RMSE = ', num2str(result_RMSE(2))]);
disp(['All Data RMSE = ', num2str(result_RMSE(3))]);

%% Drawing
figure;
semilogy(LearnCurve, 'linewidth', 1.5);
grid on; axis tight;
title('Learning Curve');
xlabel('x'); ylabel('RMSE');

figure;
Date = (1:nInput) + nFeature;
for i = 1:nTarget
        t1 = factor(nTarget);
        t1 = t1(1);
        t2 = nTarget/t1;
        subplot(t2, t1, i);
        temp_data = data(:, Close, i);
        plot(temp_data);
        title(filename{i, 2});
        xlabel('Date'); ylabel('Stock price');
        grid on; hold on;
        
        model_output = transpose(output(:, ceil(i/2)));
        if mod(i, 2) == 0
                model_output = imag(model_output);
        else
                model_output = real(model_output);
        end
        if strcmp(opts.input_type, 'delta')
                t = temp_data(Date - 1)';
                model_output = model_output + t;
        end
        plot(Date(1:nTraining), model_output(1:nTraining), '--');
        plot(Date(nTraining+1:end), model_output(nTraining+1:end), '--');
        axis tight; hold off;
        legend({'y', '$\hat{y}$ (Training)', '$\hat{y}$ (Testing)'}, 'Interpreter', 'latex', 'location', 'southeast');
end

figure;
for i = 1:nTarget
        subplot(t2, t1, i);
        if mod(i, 2) == 0
                e = imag(error(:, ceil(i/2)));
        else
                e = real(error(:, ceil(i/2)));
        end
        
        plot(1:nTraining, e(1:nTraining));
        hold on;
        plot(nTraining+1:nInput, e(nTraining+1:end));
        title(['The error between target and output (', filename{i, 2}, ')']);
        xlabel('date'); ylabel('error');
        grid on; axis tight;
        legend('Training', 'Testing', 'location', 'southeast');
end