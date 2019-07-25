clc; clear; close all;
%% Read Data
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
rmse =@(error) ( (error*error') / length(error) ) ^ 0.5;
filename = {'IBM.csv', 'IBM'
        'AAPL.csv', 'Apple'
        'DVMT.csv', 'DELL'
        'T.csv', 'AT&T'};
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
for i = 1:numel(feature_table)
        normal = ceil(max([max(abs(feature_table{i})), normal]));
end
for i = 1:numel(feature_table)
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
for i = 1:numel(feature_table)
        t = feature_table{i}(:, FP);
        t = reshape(t', 1, nFP, []);
        cnn_input = [cnn_input; t];
end

%% CNN Setting
opts.alpha = 1;
opts.nEpoch  = 20;
opts.fullconnect = 1;

%% CNN
stride = [2, 2];
cnn.layers={
        struct('type', 'i')
        struct('type', 'c', 'nFilter', 3, 'filtersize', [nTarget, nFP]-stride+1)
        struct('type', 'p', 'stride', stride);
        };
cnn.layers{1}.output{1} = cnn_input;


cnn = cnn_initial(cnn, opts);

%% 