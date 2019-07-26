clc; clear; close all;
clear global THETA;
%% Options
Ra = 0.3;
nFeature = 30;
nTarget = 4;
nClass = 2;

% For WOA
SearchAgents_no = 30;
Max_iter = 100;

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
load('IIM_TTGS.mat');
FP = FeatureSelection(data_table, IIM, nTarget);
nFP = length(FP);
FP = sort(FP);

%% Target (complex for classification)
target.real = zeros(nInput, nTarget*nClass);
tt = logical(target.real);
rate = target_table ./ target_data;
% ------------------------------------
condition = {'rate>=0', 'rate<0'};
for i = 1:nClass
        ind = eval(condition{i});
        tt(:, i:nClass:end) = ind;
end
target.real(tt) = 1;
% ------------------------ 
% boundary = calculateBoundary(rate, nClass);
% for i = 1:nTarget
%         for j = 1:nClass
%                 tt(:, (i-1)*nClass+j) = (rate(:, i)<boundary(j, i) & rate(:, i)>= boundary(j+1, i));
%         end
% end
% target(tt) = 1;
% ---------------------------
target.complex = target.real(:, 1:2:end) + target.real(:, 2:2:end) .* sqrt(-1);

for i = 1:nTarget
        index = (1:nClass)+(i-1)*nClass;
        [~, b] = max(target.real(:, index), [], 2);
        target.classification(:, i) = nClass - b;
end

%% classification
a = [];
comma = [];
for i = 1:nTarget
        a = [a, comma, '0:nClass-1'];
        comma = ',';
end
eval(['classify = allcomb(', a, ');']);

Target.real = zeros(size(target.classification));
for i = 1:size(target.classification, 1)
        for j = 1:size(classify, 1)
                if target.classification(i, :) == classify(j, :)
                        Target.real(i, j) = 1;
                end
        end
end
Target.complex = real2complex(Target.real, 2);

%% Data processing
cnn_input = feature_table(:, FP);
cnn_input = reshape(cnn_input', sqrt(nFP), sqrt(nFP), []);

%% CNN structure
cnn.layers={
        struct('type', 'i')                                                             % 19-by-19
        struct('type', 'c', 'nFilter', 12, 'filtersize', [2, 2])         % 18-by-18
        struct('type', 'p', 'stride', [2, 2]);                                % 9-by-9
        struct('type', 'c', 'nFilter', 6, 'filtersize', [2, 2])         % 8-by-8
        struct('type', 'p', 'stride', [2, 2]);                                % 4-by-4
        };
cnn.layers{1}.output{1} = cnn_input;
cnn  = cnn_initial(cnn);
% load('cnn');

nParameter_cnn = max(cnn.bias_index);
% 先弄一個Whale用來做Subclust和Block Selection
whale = rand(1, nParameter_cnn) .* 2 - 1;
cnn = cnn_fp(cnn, whale);
nDim = size(cnn.output, 1);

%% Subtractive Clustering
h = Subclustering(cnn.output(:, 1:nTraining), Ra);

%% Construct Matrix
t = []; comma = [];
for i = 1:nDim
        t = [ t, comma, '1:length(h{', int2str(i), '}.center)'];
        comma = ', ';
end
eval(['C = allcomb(', t, ');']);

%% Block Selection
C = BlockSelection(C, cnn.output(:, 1:nTraining), h, nTarget, nClass);

%% Rule index
[temp_index, para] = ConIndex(C, 0, h);
premise_index = temp_index + nParameter_cnn;
nParameter_premise = nParameter_cnn + numel(para);
nSphere_theta = nClass^nTarget - 1;
sphere_index = nParameter_premise + (1:nSphere_theta);
nParameter = nParameter_premise + nSphere_theta;

%% Initial position
whale = zeros(SearchAgents_no, nParameter);
whale(:, 1:nParameter_cnn) = rand(SearchAgents_no, nParameter_cnn) .* 2 - 1;
whale(:, nParameter_cnn+1:nParameter_premise) = para + randn(SearchAgents_no, length(para));
whale(:, nParameter_premise+1:nParameter) = ones(SearchAgents_no, nSphere_theta);

%% Training
fobj =@ model;
opts.cnn = cnn;
opts.index_rule = premise_index;
opts.index_sphere = sphere_index;
opts.C = C;
opts.nTraining = nTraining;
opts.target = Target.complex;
opts.nTarget = nTarget;
opts.nClass = nClass;
global THETA;
nRule = size(C, 1);
THETA = zeros(nRule .* (nDim+1), nClass^nTarget/2);

tic;
[Leader_score, Leader_pos, Learning_curve] = new_WOA(SearchAgents_no, Max_iter, whale, nParameter, fobj, opts);
disp(toc);

%% Testing
RMSE = fobj(Leader_pos, opts);
[result_RMSE, output] = fobj(Leader_pos, opts, 0);     % 不需要再做RLSE

%% Error rate
[~, t] = max(Target.real, [], 2);
[~, o] = max(output.real, [], 2);

target.classification = classify(t, :);
output.classification = classify(o, :);

error_index = target.classification ~= output.classification;
result_error_rate = calculateErrorRate(error_index, nInput, nTraining);
result = mean(result_error_rate);

disp(['error rate = ', num2str(result(1))]);
disp(['correct rate = ', num2str(result(2))]);
disp(['error rate(training) = ', num2str(result(3))]);
disp(['correct rate(training) = ', num2str(result(4))]);
disp(['error rate(testing) = ', num2str(result(5))]);
disp(['correct rate(testing) = ', num2str(result(6))]);


% error_index = target.classification ~= output.classification;
% nError = sum(error_index);
% nError_train = sum(error_index(1:nTraining));
% nError_test = sum(error_index(nTraining+1:end));
% disp(['error rate = ', num2str( nError/nInput )]);
% disp(['correct rate = ', num2str( 1 - nError/nInput )]);
% disp(['error rate(training) = ', num2str( nError_train / nTraining )]);
% disp(['correct rate(training) = ', num2str( 1 - nError_train / nTraining )]);
% disp(['error rate(testing) = ', num2str( nError_test / (nInput-nTraining) )]);
% disp(['correct rate(testing) = ', num2str( 1 - nError_test / (nInput-nTraining) )]);

%% Drawing
figure;
plot(Learning_curve, 'linewidth', 1.5);
title('Learning curve');
xlabel('iteration');
ylabel('RMSE');
grid on;

figure('Position', [5 50 1000 730]);
for i = 1:nTarget
        subplot(2, 2, i);
        c = find(~error_index(:, i));
        plot(c, target.classification(c, i), 'r*');
        
        hold on;
        e = find(error_index(:, i));
        plot(e, target.classification(e, i), 'bo');
        
        L = line([nTraining, nTraining], [-1, nClass]);
        set(L, 'LineStyle', '--');
        
        title([filename{i, 2}, ' (error rate = ', num2str(round(result_error_rate(i, 1)*100, 2)), '%)']);
        xlabel('Date'); ylabel('classification');
        legend({'target (correct)', 'target (mistake)'});
        axis([-inf, inf, -1, nClass]);
        set(gca, 'ytick',-1:nClass);
        set(gca, 'yticklabel', {'', '下跌', '上漲', ''});
        grid on;
end