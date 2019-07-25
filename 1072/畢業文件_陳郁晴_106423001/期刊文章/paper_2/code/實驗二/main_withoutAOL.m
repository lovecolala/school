% Different algorithm
clc; clear; close all;
model_struct = 'ANN-SCNFS';

nFeatureDate = 30;
Ra = 0.2;

nExp = 20;
iteration = 500;

gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
mape =@(target, output) mean(abs((target-output)./target));
mae =@(error) mean(abs(error));

%% Dataset
nStock = 4;
nTarget = 4;
% Get input_table and target_table
stockname = {'KOSPI', 'Nikkei 225', 'HSI', 'TAIEX'};
load('dataset');

dataset.input = [];
for i = 1:nStock
        t = input_table{i}{:, 2:end};
        t = t ./ max(abs(t));
        dataset.input = [dataset.input, t];
end

dataset.output = [];
for i = 1:nTarget
        t = target_table{i}{:, end};
        normal(i) = max(abs(t));
        t = t ./ normal(i);
        dataset.output = [dataset.output, t];
end

[fs.nInput, fs.nDim] = size(dataset.input);
fs.input = dataset.input(2:end, :) - dataset.input(1:end-1, :);
fs.target = dataset.output(2:end, :) - dataset.output(1:end-1, :);

%% Feature selection
[fs.feature_table, fs.target_table] = construct_feature(fs.input, fs.target, nFeatureDate);
nTraining = ceil(size(fs.feature_table, 1) * 0.8);

fs.data_table = [fs.feature_table, fs.target_table];
load('IIM');

fs.FP = FeatureSelection(fs.data_table, IIM, nTarget);
fs.FP = sort(fs.FP);
nFP = length(fs.FP);

fs.output = fs.feature_table(:, fs.FP);
nInput = size(fs.output, 1);

%% Transfer to target
nOutput = ceil(nTarget/2);
target = real2complex(fs.target_table, 2);

%% NN
nn.input = fs.output;
nn.nNeuro = 4;
accu_Para = 0;

% weight
nPara = nFP * nn.nNeuro;
nn.weight = randn(nFP, nn.nNeuro);
algoindex.nn.weight = reshape(1:nPara, nFP, nn.nNeuro) + accu_Para;
accu_Para = accu_Para + nPara;

% bias
nPara = nn.nNeuro;
nn.bias = zeros(1, nn.nNeuro);
algoindex.nn.bias = (1:nn.nNeuro) + accu_Para;
accu_Para = accu_Para + nPara;

% initial position
position_org = [nn.weight(:)', nn.bias];
nn = nn_fp(nn);

%% Rule in premise layer
h = Subclustering(nn.output(:, 1:nTraining), Ra);
t = []; comma = [];
for i = 1:nn.nNeuro
        t = [ t, comma, '1:length(h{', int2str(i), '}.center)'];
        comma = ', ';
end
eval(['C = allcomb(', t, ');']);
C = BlockSelection(C, nn.output(:, 1:nTraining), h, nTarget, 10);
[nRule, nDim] = size(C);

%% SCFS
[index_rule, para] = ConIndex(C, 0, h);

% Center and Sigma
nPara = numel(para);
algoindex.cs = index_rule + accu_Para;
accu_Para = accu_Para + nPara;
position_org = [position_org, para+randn(size(para))];

% Lambda in SCFS
nPara = nTarget - 1;
algoindex.lambda = (1:nPara) + accu_Para;
accu_Para = accu_Para + nPara;
position_org = [position_org, ones(1, nPara)];

%% Aim object
% [aol, nThen] = getaol(target(1:nTraining, :), 0.6, nTarget);

%% RLSE
nPara = (1+nDim)*nRule*nOutput;
algoindex.theta = reshape((1:nPara), (1+nDim)*nRule, nOutput) + accu_Para;
accu_Para = accu_Para + nPara;
position_rlse = randn(1, (1+nDim)*nRule*nOutput);

%% Option of model
fobj =@ model_withoutAOL;

opts.nn = nn;
opts.algoindex = algoindex;
opts.C = C;
opts.nTraining = nTraining;
opts.target = target;
opts.nTarget = nTarget;

%% Experience
[algoname, algofunc, algoopts] = algorithm(6);
if algoopts.rlse == 1
        position = position_org;
elseif algoopts.rlse == 0
        position = [position_org, position_rlse];
end

disp(['algorithm = ', algoname]);
exp_result = cell(1, nExp);
result_RMSE = zeros(1, nExp);
waste_time = zeros(1, nExp);
for e = 1:nExp
        fprintf('---------- experience %d ......', e);
        
        %% Training
        tic;
        [Leader_score, Leader_pos, Learning_curve] =...
                algofunc(iteration, position, fobj, algoopts, opts);
        waste_time(e) = toc;
        
        exp_result{e}.Leader_score = Leader_score;
        exp_result{e}.Leader_pos = Leader_pos;
        exp_result{e}.Learning_curve = Learning_curve;
        
        %% Testing
        if algoopts.rlse == 0
                [result_RMSE(e), output] = fobj(Leader_pos, opts, algoopts.rlse, 0);
                
        elseif algoopts.rlse == 1
                [~, ~, theta] = fobj(Leader_pos, opts, algoopts.rlse);
                [result_RMSE(e), output] = fobj(Leader_pos, opts, algoopts.rlse, 0, theta);
                exp_result{e}.theta = theta;
                
        end
        exp_result{e}.output = output;
        
        %% Figure
        figure('Name', ['exp', num2str(e), '_Learning Curve']);
        plot(Learning_curve);
        title(['Learning Curve (algo: ', algoname, ')']);
        xlabel('iteration'); ylabel('RMSE');
        grid on;
        
        o = complex2real(output, 2, nTarget) .* normal;
        t = complex2real(target, 2, nTarget) .* normal;
        error = t - o;
        
        exp_result{e}.rmse = [rms(error, 1); rms(error(1:nTraining, :), 1); rms(error(nTraining+1:end, :), 1)].';
        exp_result{e}.mape = [mape(t, o); mape(t(1:nTraining, :), o(1:nTraining, :)); mape(t(nTraining+1:end, :), o(nTraining+1:end, :))].';
        exp_result{e}.mae = [mae(error); mae(error(1:nTraining, :)); mae(error(nTraining+1:end, :))].';
        
        date = target_table{1}{:, 1};
        for i = 1:nTarget
                figure('Name', ['exp', int2str(e), '_', stockname{i}, '_result']);
                plot(date, target_table{i}{:, end}, 'b-');
                hold on;
                
                ind = (1:nInput) + nFeatureDate;
                pre_result = target_table{i}{ind, end} + o(:, i);
                plot(date(ind+1), pre_result, 'r--');
                
                y_min = min([target_table{i}{:, end}; pre_result]);
                y_max = max([target_table{i}{:, end}; pre_result]) * 25/24;
                ylim([y_min, y_max]);
                
                line(date([nTraining, nTraining]+nFeatureDate), get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k');
                text(date(ceil(nTraining/2) + nFeatureDate), y_max*49/50, {'Training'; 'phase'}, ...
                        'HorizontalAlignment', 'center', 'color', 'k');
                text(date(ceil((nTraining+nInput)/2) + nFeatureDate), y_max*49/50, {'Testing'; 'phase'}, ...
                        'HorizontalAlignment', 'center', 'color', 'k');
                
                title(stockname{i});
                xlabel('Trading date index'); ylabel('Close price');
                legend('Target', 'Forecast', 'location', 'southwest');
                grid on; hold off;
                
                % -------------------------------
                
                figure('Name', ['exp', int2str(e), '_', stockname{i}, '_error']);
                bar(date(ind+1), error(:, i));
                title(stockname{i});
                xlabel('Trading date index'); ylabel('Prediction error');
                grid on; hold on;
                
                y_min = min(error(:, i));
                y_max = max(error(:, i));
                y_max = y_max + (y_max-y_min) * 1/5;
                xlim(['1-Jan-2018', date(end)]);
                ylim([y_min, y_max]);
                
                line(date([nTraining, nTraining]+nFeatureDate), get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k');
                text(date(ceil(nTraining/2) + nFeatureDate), y_max*9/10, {'Training'; 'phase'}, ...
                        'HorizontalAlignment', 'center', 'color', 'k');
                text(date(ceil((nTraining+nInput)/2) + nFeatureDate), y_max*9/10, {'Testing'; 'phase'}, ...
                        'HorizontalAlignment', 'center', 'color', 'k');
                
        end
        disp('done');
end
F = findall(0, 'type', 'figure');
savefig(F, ['result_', model_struct]);

[~, min_idx] = min(result_RMSE);
close(1:(nTarget*2+1)*(min_idx-1));
close((nTarget*2+1)*min_idx+1:(nTarget*2+1)*nExp);
F = findall(0, 'type', 'figure');
savefig(F, ['result_', model_struct, '_thebest']);

close all;
save(['result_', model_struct]);