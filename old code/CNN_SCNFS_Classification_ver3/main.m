clc; clear; close all;
clear global THETA;
%% Options
Ra = 0.3;
nFeature = 30;
nTarget = 4;
nClass = 5;

% For WOA
SearchAgents_no = 30;
Max_iter = 100;

nFP = 361;
cnn_input_size = sqrt(nFP);

%% Get Stock Data
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;

filename = {'2330.TW.csv', 'tsmc'
        '^TWII.csv', 'TAIEX'
        'GOOGL.csv', 'Google'
        '^GSPC.csv', 'S&P 500'};
Stockdata = cell(1, nTarget);
for i = 1:nTarget
        Stockdata{i} = csvread(filename{i, 1}, 1, 0);
end

%% Intersect the Date
% 先得到會有哪此Date是交集的
date = Stockdata{1}(:, 1);
for i = 2:nTarget
        date = intersect(date, Stockdata{i}(:, 1));
end
nData = length(date) - 1;
nInput = nData - nFeature;
nTraining = ceil(nInput*0.9);
index = repmat(0:nFeature, nInput, 1) + (1:nInput)';
index_feature = index(:, 1:end-1);
index_target = index(:, end);

%% Feature Table
feature_table = [];
target_table = [];
target_data = [];
Close = 4;
data = cell(1, nTarget);
for i = 1:nTarget
        % 將股票資料與交集的date做交集，得知符合的是哪一些index
        [~, t] = intersect(Stockdata{i}(:, 1), date);
        % 有包含Volumn(交易量) ↓
        data{i} = Stockdata{i}(t, 2:end);
        delta = data{i}(2:end, :) - data{i}(1:end-1, :);
        for j = 1:size(delta, 2)
                dd = delta(:, j);
                feature_table = [feature_table, dd(index_feature)];
                if j == Close
                        target_table = [target_table, dd(index_target)];
                        target_data = [target_data, data{i}(nFeature+1:end-1, j)];
                end
        end
end

%% Feature Selection
data_table = [feature_table, target_table];
% FP = FeatureSelection(data_table, nTarget);
load('FP_TTGS.mat');
FP = FP(1:nFP);
FP = sort(FP);

%% Target (complex for classification)
target = zeros(nInput, nTarget*nClass);
tt = logical(target);
rate = target_table ./ target_data;
% ------------------------ 
condition = {'rate>0.02', 'rate<=0.02 & rate>0', 'rate==0', 'rate<0 & rate>-0.02', 'rate<-0.02'};
for i = 1:nClass
        ind = eval(condition{i});
        tt(:, i:nClass:end) = ind;
end
target(tt) = 1;
% ------------------------ 
% boundary = calculateBoundary(rate, nClass);
% for i = 1:nTarget
%         for j = 1:nClass
%                 tt(:, (i-1)*nClass+j) = (rate(:, i)>boundary(j, i) & rate(:, i)<= boundary(j+1, i));
%         end
% end
% target(tt) = 1;
% ---------------------------

%% Data processing
cnn_input = feature_table(:, FP);
cnn_input = reshape(cnn_input', cnn_input_size, nFP/cnn_input_size, []);

%% CNN structure
% cnn.layers={
%         struct('type', 'i')                                                             % 19-by-19
%         struct('type', 'c', 'nFilter', 1, 'filtersize', [2, 2])         % 18-by-18
%         struct('type', 'p', 'stride', [2, 2]);                                % 9-by-9
%         struct('type', 'c', 'nFilter', 1, 'filtersize', [2, 2])         % 8-by-8 
%         struct('type', 'p', 'stride', [2, 2]);                                % 4-by-4
%         struct('type', 'c', 'nFilter', 1, 'filtersize', [3, 3])         % 2-by-2
%         };
% cnn.layers{1}.output{1} = cnn_input;
% cnn  = cnn_initial(cnn);
load('cnn');

for i = numel(cnn.layers):-1:1
        if strcmp(cnn.layers{i}.type, 'c')
                nParameter_cnn = max(cnn.layers{i}.filter_index{end}(:));
                break;
        end
end
% 先弄一個Whale用來做Subclust和Block Selection
% whale = rand(1, nParameter_cnn) .* 2 - 1;
% cnn = cnn_fp(cnn, whale);

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
C = BlockSelection(C, CnnOutput(:, 1:nTraining), h, nTarget, nClass);

%% Rule index
[temp_index, para] = ConIndex(C, 0, h);
premise_index = temp_index + nParameter_cnn;
nParameter_premise = nParameter_cnn + numel(para);
sphere_index = nParameter_premise + (1:nTarget*nClass-1);
nParameter = nParameter_premise + nTarget*nClass;

%% Initial position
whale = zeros(SearchAgents_no, nParameter);
whale(:, 1:nParameter_cnn) = rand(SearchAgents_no, nParameter_cnn) .* 2 - 1;
whale(:, nParameter_cnn+1:nParameter_premise) = para + randn(SearchAgents_no, length(para));
whale(:, nParameter_premise+1:nParameter) = ones(SearchAgents_no, nTarget*nClass);

%% Training
fobj =@ model;
opts.cnn = cnn;
opts.premise_index = premise_index;
opts.sphere_index = sphere_index;
opts.C = C;
opts.nTraining = nTraining;
opts.target = target;
opts.nTarget = nTarget;
opts.nClass = nClass;
global THETA;
nRule = size(C, 1);
THETA = zeros(nRule .* (nDim+1), nTarget*nClass);

tic;
[Leader_score, Leader_pos, Learning_curve] = new_WOA(SearchAgents_no, Max_iter, whale, nParameter, fobj, opts);
disp(toc);

%% Testing
RMSE = model(Leader_pos, opts);
[result_RMSE, output] = model(Leader_pos, opts, 0);

%% Error rate
output_classification = zeros(nInput, nTarget);
target_classification = output_classification;
for i = 1:nTarget
        index = (1:nClass)+(i-1)*nClass;
        [~, b] = max(output(:, index), [], 2);
        b = mod(b, nClass);
        output_classification(:, i) = b;
        
        [~, b] = max(target(:, index), [], 2);
        b = mod((b-1), nClass)+1;
        target_classification(:, i) = b;
end

error_index = (target_classification ~= output_classification);
sum_error = sum(error_index);
error_rate = sum(sum_error) ./ (nInput .* nTarget);
disp(['error rate = ', num2str(error_rate)]);
disp(['correct rate = ', num2str(1-error_rate)]);

error_rate_train = sum(sum(error_index(1:nTraining, :))) ./ (nTraining .* nTarget);
disp(['error rate(training) = ', num2str(error_rate_train)]);
disp(['correct rate(training) = ', num2str(1-error_rate_train)]);

error_rate_test = sum(sum(error_index(nTraining+1:end, :))) ./ ((nInput-nTraining) .* nTarget);
disp(['error rate(training) = ', num2str(error_rate_test)]);
disp(['correct rate(training) = ', num2str(1-error_rate_test)]);

%% Drawing
figure;
plot(Learning_curve, 'linewidth', 1.5);
title('Learning curve');
xlabel('iteration');
ylabel('RMSE');
grid on;

figure;
for i = 1:nTarget
        subplot(2, 2, i);
        c = find(~error_index(:, i));
        plot(c, target_classification(c, i), 'r*');
        hold on;
        e = find(error_index(:, i));
        plot(e, target_classification(e, i), 'bo');
        
        error_stock = sum_error(i) ./ nInput;
        title([filename{i, 2}, ' (error rate = ', num2str(round(error_stock*100, 2)), '%)']);
        xlabel('Date'); ylabel('classification');
        legend('target (correct)', 'target (mistake)');
        axis([-inf, inf, 0, 6]);
        set(gca, 'ytick',0:6);
        set(gca, 'yticklabel', {'', '大跌', '小跌', '平盤', '小漲', '大漲', ''});
        grid on;
end