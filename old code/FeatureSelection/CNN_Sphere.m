clc; clear; close all;
%% Read Data
filename = 'IBM.csv';
Close = 4;
StockData = csvread(filename, 1, Close);
StockData = StockData(:, 1);
normal = max([abs(StockData), 1]);
StockData = StockData ./ normal;

%% Feature Selection
nFeature = 30;
nTarget = size(StockData, 2);
[feature_table, FP] = FeatureSelection(StockData, nFeature, nTarget);
FP = sort(FP);

%% Data processing
input_stockprice = feature_table{1}(:, FP);

nInput = size(input_stockprice, 1);
nFP = size(input_stockprice, 2);
t = factor(nFP);
nColumn = t(end);
nRow = nFP / nColumn;

input = zeros(nRow, nColumn, nInput);

for i = 1:nInput
        t = input_stockprice(i, :);
        t = reshape(t, nColumn, nRow)';
        input(:, :, i) = t;
end

%% Ant Setting
opts.nAnt = 10;
opts.nNewAnt  = 20;
opts.iteration = 100;

opts.blockselect = 1;
opts.fullconnect = 0;

%% Convolutional Neural Network
cnn.layers={
        struct('type', 'i')                                                             % Input Layer
        struct('type', 'c', 'nFilter', 3, 'filtersize', [2, 3])                  % Convolution Layer, filter = 2*3, filter個數=3
        struct('type', 'p', 'stride', [1, 3])                                                 % Pooling Layer, 從1*2取mean值
        };
cnn.layers{1}.output{1} = input;

% Filter的初始（Ant）
cnn = cnn_initial(cnn, opts);
cnn = cnn_fp(cnn, opts);

%% reshape CNN
nDim = numel(cnn.layers{end}.output);
CnnOutput = zeros(nDim, nInput);
for i = 1:nDim
        CnnOutput(i, :) = cat(1, [], cnn.layers{end}.output{i}(:))';
end

%% Subtractive Clustering
% Center and Sigma
h = Subclustering(CnnOutput, 0.3);

%% Construct Matrix
t = []; comma = [];
for i = 1:nDim
        t = [ t, comma, '1:length(h{i}.center)'];
        comma = ', ';
end
eval(['C_original = allcomb(', t, ');']);

%% Block Selection
if opts.blockselect
        C = BlockSelection(C_original, CnnOutput, h, nTarget);
else
        C = C_original;
end

%% construct index and parameter
[Rule_index, para] = ConIndex(C, 0, h);

%% Initial Ant


%% Training


