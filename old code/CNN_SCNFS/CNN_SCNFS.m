clc; clear; close all;
%% Option
opts.nAnt = 5;
opts.new_nAnt = 10;
opts.eva_rate = 0.9;
opts.learning_rate = 0.1;

nTarget = 1;
Ra = 0.3;
iteration = 100;

%% Read Data
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
rmse =@(error) ( (error*error') / length(error) ) ^ 0.5;
filename = {'IBM.csv', 'IBM'
        'DVMT.csv', 'DELL'
        'AAPL.csv', 'Apple'
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
normal = ceil(max([max(feature_table), 1]));

%% Target
nOutput = ceil(nTarget/2);
nInput = size(feature_table, 1);
target = zeros(nOutput, nInput);
feature_table = feature_table ./ normal;
for i = 1:nTarget
        t = feature_table(:, end-nTarget+i)';
        if mod(i, 2) == 0
                t = t .* sqrt(-1);
        end
        target(ceil(i/2), :) = target(ceil(i/2), :) + t;
end

%% Data processing
nFP = length(FP);
cnn_input = feature_table(:, FP);
if mod(nFP, 4)
        cnn_input = reshape(cnn_input', 4, nFP/4, []);
else
        cnn_input = reshape(cnn_input', 2, nFP/2, []);
end

%% Set the parameter
weight = gauF(1:opts.nAnt, 1, opts.learning_rate*opts.nAnt) ./ (opts.learning_rate*opts.nAnt*sqrt(2*pi));
opts.prob = weight ./ sum(weight);

opts.blockselect = 1;

%% CNN
sizeInput = [size(cnn_input, 1), size(cnn_input, 2)];
stride = [2, 2];
cnn.layers={
        struct('type', 'i')
        struct('type', 'c', 'nFilter', 3, 'filtersize', sizeInput-stride+1)
        struct('type', 'p', 'stride', stride)
        };

cnn.layers{1}.output{1} = cnn_input;

[swarm, cnn] = cnn_initial(cnn, opts);
cnn = cnn_fp( cnn, swarm, ones(1, numel(swarm)) );

%% reshape CNN
% 到SCNFS的維度數，也就是CNN的output數 (3)
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
        t = [ t, comma, '1:length(h{', int2str(i), '}.center)'];
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
swarm(end+1).ant(1).position = para;
for i = 2:opts.nAnt
        swarm(end).ant(i).position = para .* rand ./ 2;
end

%% Training
all_nAnt = opts.nAnt + opts.new_nAnt;
nRule = size(C, 1);
theta = zeros(nRule.*(nDim+1), nOutput);
LearnCurve = [];

for i = 1:iteration
        swarm = antrenew(swarm, opts);
        for j = 1:numel(swarm)
                for k = 1:all_nAnt
                        %% CNN
                        index = ones(1, numel(swarm));
                        index(j) = k;
                        cnn = cnn_fp( cnn, swarm, index );
                        for q = 1:nDim
                                CnnOutput(q, :) = cat(1, [], cnn.layers{end}.output{q}(:))';
                        end
                        
                        %% Beta
                        % beta = [nRule, nInput, nOutput]
                        for q = 1:nRule
                                Center = Rule_index.center(q, :);
                                CenterSigma = swarm(end).ant(index(end)).position([Center', (Center+1)']);
                                beta(q, :, :) = SphereCom(CnnOutput, CenterSigma, nTarget);
                        end
                        
                        %% RLSE
                        A = repmat([ones(nInput, 1), CnnOutput'], 1, nRule);
                        for q = 1:nOutput
                                A_beta = A .* repelem( transpose(beta(:, :, q)), 1, 1+nDim );
                                theta(:, q) = RLSE(A_beta, target(q, :), theta(:, q));
                                t = A_beta * theta(:, q);
                                output(q, :) = transpose(t);
                        end
                        
                        %% RMSE
                        error = target - output;
                        RMSE(k) = rmse(transpose(error(:)));
                end
                %% Sorting
                [RMSE, sorting] = sort(RMSE);
                sorting = sorting(1:opts.nAnt);
                swarm(j).ant = swarm(j).ant(sorting);
                
                LearnCurve = [LearnCurve, RMSE(1)];
                
        end
end

%% Drawing
% Learning Curve
figure;
semilogy(LearnCurve .* normal.^2);
grid on; axis tight;
title('Learning Curve');
xlabel('x'); ylabel('RMSE');

% Stock
figure('Name', 'CNN+SCNFS', 'NumberTitle', 'off');
date = nFeature+1 : size(StockData, 1);
for i = 1:nTarget
        t1 = factor(nTarget);
        t1 = t1(1);
        t2 = nTarget / t1;
        subplot(t2, t1, i);
        plot(StockData(:, i), 'linewidth', 1.5);
        title(filename{i, 2});
        xlabel('Date'); ylabel('Stock price');
        hold on; 
        
        model_output = zeros(size(StockData, 1)-nFeature, 1);
        model_output(1) = StockData(nFeature+1, i);
        t = output(ceil(i/2), :);
        if mod(i, 2) == 0
                t = imag(t);
        else
                t = real(t);
        end
        model_output(2:end) = t'.*normal;
        model_output = cumsum(model_output);
        plot(date, model_output, '--', 'linewidth', 1.5);
        grid on; axis tight;
        hold off;
end
