clc; clear; close all;
%% Read Data
filename = {'IBM.csv', 'IBM'
        'AAPL.csv', 'Apple'
        'DVMT.csv', 'DELL'
        'T.csv', 'AT&T'};
% filename = 'IBM.csv';
Close = 4;
StockData = [];
nTarget = size(filename, 1);
for i = 1:nTarget
        t = csvread(filename{i, 1}, 1, Close);
        t = t(:, 1);
        StockData = [StockData, t];
end

%% Feature Selection
nFeature = 30;
[feature_table, FP] = FeatureSelection(StockData, nFeature, nTarget);
FP = sort(FP);
normal = 1;
for i = 1:nTarget
        normal = ceil(max([max(abs(feature_table{i})), normal]));
end
for i = 1:nTarget
        feature_table{i} = feature_table{i} ./ normal;
end
nInput = size(feature_table{1}, 1);


%% Target
target = zeros(nTarget, nInput);
for i = 1:nTarget
        target(i, :) = feature_table{i}(:, end)';
end

%% Data processing
cnn_input = [];
nFP = length(FP);
if nTarget == 1
        t = feature_table{1}(:, FP);
        t = reshape(t', 2, nFP/2, []);
        cnn_input = [cnn_input ; t];
else
        for i = 1:nTarget
                t = feature_table{i}(:, FP);
                t = reshape(t', 1, nFP, []);
                cnn_input = [cnn_input ; t];
        end
end

%% CNN Setting
opts.alpha = 0.5;
opts.nEpoch  = 100;
opts.fullconnect = 1;

%% Convolutional Neural Network
sizeInput = [size(cnn_input, 1), size(cnn_input, 2)];
stride = [2, 2];
cnn.layers={
        struct('type', 'i')                                                             % Input Layer
        struct('type', 'c', 'nFilter', 3, 'filtersize', sizeInput-stride+1)                  % Convolution Layer, filter = 2*3, filter個數=3
        struct('type', 'p', 'stride', stride)                                                 % Pooling Layer, 從1*2取mean值
        };
cnn.layers{1}.output{1} = cnn_input;

% Filter的初始（Ant）
cnn = cnn_initial(cnn, opts, nTarget);
cnn = cnn_train(cnn, target, opts);

%% draw
cnn = cnn_fp(cnn, opts);
output = cnn.output;

figure('Name', 'CNN', 'NumberTitle', 'off');
date = nFeature+1 : size(StockData, 1);
for i = 1:nTarget
        t1 = factor(nTarget);
        t1 = t1(end);
        t2 = nTarget / t1;
        subplot(t2, t1, i);
        plot(StockData(:, i), 'linewidth', 1.5);
        title(filename{i, 2});
        xlabel('Date'); ylabel('Stock price');
        hold on;
        
        model_output = zeros(size(StockData, 1)-nFeature, 1);
        model_output(1) = StockData(nFeature+1, i);
        model_output(2:end) = output(i, :) .* normal;
        model_output = cumsum(model_output);
        plot(date, model_output, '--', 'linewidth', 1.5);
        grid on; axis tight;
        hold off;
end

figure; 
semilogy(cnn.LearnCurve.*normal^2, 'linewidth', 1.5);
grid on;
title('Learning Curve'); xlabel('iteration'); ylabel('MSE');