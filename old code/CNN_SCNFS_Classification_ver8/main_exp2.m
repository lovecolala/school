clc; clear; close all;
clear global THETA;

%% Options
Ra = 0.5;
nFeatureDate = 30;
nStock = 4;
nTarget = 1;
nClass = 2;
nExp = 15;

nWhale = 30;
Max_iter = 100;

%% Function
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;

%% 眔Data Set
filename = {'2330.TW.csv', 'tsmc'
        '^TWII.csv', 'TAIEX'
        'GOOGL.csv', 'Google'
        '^GSPC.csv', 'S&P 500'};
dataset = cell(1, nStock);
for i = 1:nStock
        dataset{i} = csvread(filename{i, 1}, 1, 0);
end

%% Τユ栋ら戳 Intersect
IntersectDate = dataset{1}(:, 1);
for i = 2:nStock
        IntersectDate = intersect(IntersectDate, dataset{i}(:, 1));
end
nInput = length(IntersectDate);
nAttribute = length(dataset{1}(1, 2:end));

%% ユ栋らData Set
nDim = nAttribute*nStock;
input.price = zeros(nInput, nDim);
for i = 1:nStock
        temp = (1:nAttribute)+(i-1)*nAttribute;
        [~, index_intersect] = intersect(dataset{i}(:, 1), IntersectDate);
        input.price(:, temp) = dataset{i}(index_intersect, 2:end);
end
input.delta = input.price(2:end, :) - input.price(1:end-1, :);

%% Feature Table
nFeature = nFeatureDate * nDim;
nInput_delta = nInput - nFeatureDate - 1;
nTraining = ceil(nInput_delta*0.9);
feature_table = zeros(nInput_delta, nFeatureDate * nStock);
target_table = zeros(nInput_delta, nTarget);
index_target = 4;       % Μ絃基い抖

index_getfeature = (0:nFeatureDate) + (1:nInput_delta)';
index_feature_table = index_getfeature(:, 1:end-1);
index_target_table = index_getfeature(:, end);

for i = 1:size(input.delta, 2)
        temp = (1:nFeatureDate) + (i-1)*nFeatureDate;
        dd = input.delta(:, i);
        feature_table(:, temp) = dd(index_feature_table);
end

for i = 1:nTarget
        ii = (i-1)*nAttribute + index_target;
        dd = input.delta(:, ii);
        target_table(:, i) = dd(index_target_table);
end

%% Feature Selection
data_table = [feature_table, target_table];
% IIM = CalculateIIM(data_table);
load('IIM_exp2');
FP = FeatureSelection(data_table, IIM, nTarget);
nFP = length(FP);
FP = sort(FP);

%% Model Target
condition = {'target_table < 0', 'target_table >= 0'};
target.real = zeros(nInput_delta, nTarget*nClass);
tt = logical(target.real);
for i = 1:nClass
        ind = eval(condition{i});
        tt(:, i:nClass:end) = ind;
end
target.real(tt) = 1;
target.complex = real2complex(target.real, 2);
target = real2class(target, nClass, nTarget);

%% CNN Structure
nLabel = 4;             % Fully-connected layer  neuro 计
cnn.layers = {
        struct('type', 'i')
        struct('type', 'c', 'nFilter', 12, 'filtersize', [2, 2])
        struct('type', 'p', 'stride', [2, 2])
        struct('type', 'c', 'nFilter', 6, 'filtersize', [2, 2])
        struct('type', 'p', 'stride', [2, 2])
        };
cnn.layers{1}.output{1} = reshape(feature_table(:, FP)', sqrt(nFP), sqrt(nFP), []);
cnn = cnn_initial(cnn, nLabel);

%% ∕﹚把计秖 - 秈︽CNN FP
nParameter.cnn = max(cnn.bias_index);
whale = rand(1, nParameter.cnn) .* 2 - 1;
cnn = cnn_fp(cnn, whale);

%% ∕﹚把计秖 - Subclust and BlockSelection
h = Subclustering(cnn.output(:, 1:nTraining), Ra);

t = []; comma = [];
for i = 1:nLabel
        t = [ t, comma, '1:length(h{', int2str(i), '}.center)'];
        comma = ', ';
end
eval(['C = allcomb(', t, ');']);

C = BlockSelection(C, cnn.output(:, 1:nTraining), h, nTarget, nClass);

%% ∕﹚把计秖
[index_rule, para] = ConIndex(C, 0, h);
nParameter.CenterSigma = numel(para);
index_rule = index_rule + nParameter.cnn;

nParameter.sphere = nClass*nTarget - 1;
index_sphere = nParameter.CenterSigma + (1:nParameter.sphere);

nParameter.all = nParameter.cnn + nParameter.CenterSigma + nParameter.sphere;

%% Training
fobj =@ model;
nRule = size(C, 1);

opts.cnn = cnn;
opts.index_rule = index_rule;
opts.index_sphere = index_sphere;
opts.C = C;
opts.nTraining = nTraining;
opts.target = target;
opts.nTarget = nTarget;
opts.nClass = nClass;

%% Experience
exp_result = zeros(nExp, 6);
for e = 1:nExp
        disp(['----------- experience', num2str(e), '-----------']);
        %% initial whale position
        whale = zeros(nWhale, nParameter.all);
        whale(:, 1:nParameter.cnn) = rand(nWhale, nParameter.cnn) .* 2 - 1;
        whale(:, (1:nParameter.CenterSigma)+nParameter.cnn) = para + randn(nWhale, length(para));
        whale(:, (1:nParameter.sphere) + end - nParameter.sphere) = ones(nWhale, nParameter.sphere);
        
        %% Training
        tic;
        [Learder_score, Leader_pos, Learning_curve] = new_WOA(nWhale, Max_iter, whale, nParameter.all, fobj, opts);
        toc;
        
        %% Testing
        [result_RMSE, output] = fobj(Leader_pos, opts, 0);
        
        %% Classification
        output.real = complex2real(output.complex, 2);
        output = real2class(output, nClass, nTarget);
        index_error = target.classification ~= output.classification;
        result_error_rate = calculateErrorRate(index_error, nInput_delta, nTraining);
        result = mean(result_error_rate, 1);
        
        disp(['error rate = ', num2str(result(1))]);
        disp(['correct rate = ', num2str(result(2))]);
        disp(['error rate(training) = ', num2str(result(3))]);
        disp(['correct rate(training) = ', num2str(result(4))]);
        disp(['error rate(testing) = ', num2str(result(5))]);
        disp(['correct rate(testing) = ', num2str(result(6))]);
        
        exp_result(e, :) = result;
        
        %% Drawing
        figure('Name', ['exp: ', num2str(e), '_Learning curve'],'NumberTitle','off');
        plot(Learning_curve, 'linewidth', 1.5);
        title('Learning curve');
        xlabel('iteration'); ylabel('performance');
        grid on;
        
        figure('Name', ['exp: ', num2str(e), '_result'],'NumberTitle','off');
        t = 1;
        for i = 1:nTarget
                subplot(t, nTarget/t, i);
                plot(target.classification(:,i), 'r.');
                hold on;
                plot(output.classification(:, i), 'b.');
                
                L = line([nTraining, nTraining], [0, nClass+1]);
                set(L, 'LineStyle', '--');
                
                title([filename{i, 2}, ' (error rate = ', num2str(round(result_error_rate(i, 1)*100, 2)), '%)']);
                xlabel('date'); ylabel('classification');
                legend('target classification', 'output classification');
                axis([-inf, inf, 0, nClass+1]);
                set(gca, 'ytick', 0:nClass+1);
                set(gca, 'yticklabel', {'', '禴', '害', ''});
                grid on;
        end
        
        figure('Name', ['exp: ', num2str(e), '_error'],'NumberTitle','off');
        for i = 1:nTarget
                subplot(t, nTarget/t, i);
                plot(target.classification(:, i) - output.classification(:, i));
                grid on;
                xlabel('date'); ylabel('error');
                axis([-inf, inf, -2, 2]);
                set(gca, 'ytick', -2:2);
                set(gca, 'yticklabel', {'', 'ヘ夹禴/箇代害', 'タ絋箇代', 'ヘ夹害/箇代禴', ''});
        end
end