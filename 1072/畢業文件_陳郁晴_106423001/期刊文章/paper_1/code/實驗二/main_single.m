clc; clear; close all;

stock = {'^DJI.csv', 'DJIA'
                '^GSPC.csv', 'S&P 500'
                '^IXIC.csv', 'NASDAQ Composite'
                '^NYA.csv', 'NYSE Composite'};

nFeatureDate = 30;
nn.transferfunc = '';
Ra = 0.2;

nExp = 20;
iteration = 500;

antopts.nAnt = 5;
antopts.new_nAnt = 25;
antopts.eva_rate = 0.8;
antopts.learning_rate = 0.5;

gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;
mape =@(target, output) sum(abs((target(:)-output(:))./target(:))) ./ numel(target);
mae =@(error) sum(abs(error(:))) ./ numel(error);

%% 不同的實驗數據
for exp_type = size(stock, 1)
        nStock = 1;
        nTarget = 1;
        filename.input ={
                stock{exp_type, 1}
                };
        filename.output ={
                stock{exp_type, 1}, stock{exp_type, 2}
                };
        
        %% 取得Data set
        dataset.input = [];
        for i = 1:nStock
                t = csvread(filename.input{i}, 1, 0);
                t = t(:, 2:end);
                t = t ./ max(abs(t));
                dataset.input = [dataset.input, t];
        end
        
        dataset.output =[];
        for i = 1:nTarget
                t = csvread(filename.output{i, 1}, 1, 0);
                t = t(:, end);
                normal(i) = max(abs(t));
                t = t ./ normal(i);
                dataset.output = [dataset.output, t];
        end
        
        fs.nInput = size(dataset.input, 1);
        fs.nDim = size(dataset.input, 2);
        fs.input = dataset.input(2:end, :) - dataset.input(1:end-1, :);
        fs.target = dataset.output(2:end, :) - dataset.output(1:end-1, :);
        
        %% Feature Selection
        [fs.feature_table, fs.target_table] = construct_feature(fs.input, fs.target, nFeatureDate);
        nTraining = ceil(size(fs.feature_table, 1) * 0.8);
        
        fs.data_table = [fs.feature_table, fs.target_table];
        fs.IIM = CalculateIIM(fs.data_table);
        
        fs.FP = FeatureSelection(fs.data_table, fs.IIM, nTarget);
        fs.FP = sort(fs.FP);
        nFP = length(fs.FP);
        
        fs.output = fs.feature_table(:, fs.FP);
        nInput = size(fs.output, 1);
        
        %% Construct the target data
        nOutput = ceil(nTarget/2);
        target = real2complex(fs.target_table, 2);
        
        %% NN
        nn.input = fs.output;
        nn.nNeuro = 4;
        accu_Para = 0;
        
        % weight
        nPara = nFP * nn.nNeuro;
        nn.weight = randn(nFP, nn.nNeuro);
        antindex.nn.weight = reshape(1:nPara, nFP, nn.nNeuro);
        accu_Para = accu_Para + nPara;
        
        % bias
        nPara = nn.nNeuro;
        nn.bias = zeros(1, nn.nNeuro);
        antindex.nn.bias = (1:nn.nNeuro) + accu_Para;
        accu_Para = accu_Para + nPara;
        
        % Construct original position
        org_ant = nn.weight(:)';
        org_ant = [org_ant, nn.bias];
        
        nn = nn_fp(nn, nn.transferfunc);
        
        %% Decide the number of the rule
        h = Subclustering(nn.output(:, 1:nTraining), Ra);
        t = []; comma = [];
        for i = 1:nn.nNeuro
                t = [ t, comma, '1:length(h{', int2str(i), '}.center)'];
                comma = ', ';
        end
        eval(['C = allcomb(', t, ');']);
        C = BlockSelection(C, nn.output(:, 1:nTraining), h, nTarget);
        
        %% SCFS
        [index_rule, para] = ConIndex(C, 0, h);
        
        % CenterSigma
        nPara = numel(para);
        antindex.cs = index_rule + accu_Para;
        accu_Para = accu_Para + nPara;
        org_ant = [org_ant, para+randn(size(para))];
        
        % Lambda in SCFS
        nPara = nTarget - 1;
        antindex.lambda = (1:nPara) + accu_Para;
        accu_Para = accu_Para + nPara;
        org_ant = [org_ant, ones(1, nPara)];
        
        %% Aim Object Layer
        [aol, nThen] = getaol(target(1:nTraining, :), 0.6, nTarget);
        
        %% Option of model
        fobj =@ model;
        nRule = size(C, 1);
        
        opts.nn = nn;
        opts.antindex = antindex;
        opts.aol = aol;
        opts.nThen = nThen;
        opts.C = C;
        opts.nTraining = nTraining;
        opts.target = target;
        opts.nTarget = nTarget;
        
        %% Experience
        exp_error = cell(1, nExp);
        exp_output = cell(1, nExp);
        entire_RMSE = zeros(1, nExp);
        for e = 1:nExp
                disp(['----------- experience ', num2str(e), '-----------']);
                ant = repmat(org_ant, antopts.nAnt, 1);
                ant(2:antopts.nAnt, :) = ant(2:antopts.nAnt, :) + randn(size(ant(2:antopts.nAnt, :)));
                
                %% Training
                tic;
                [Leader_score, Leader_pos, Learning_curve] = ...
                        CACO(iteration, ant, accu_Para, fobj, antopts, opts);
                toc;
                
                %% Testing
                RMSE = fobj(Leader_pos, opts);
                [result_RMSE, output] = fobj(Leader_pos, opts, 0);
                disp(['RMSE(training): ', num2str(RMSE)]);
                disp(['entire RMSE: ', num2str(result_RMSE)]);
                entire_RMSE(e) = result_RMSE;
                
                %% Figure
                figure('Name', ['exp', num2str(e), '_Learning Curve']);
                plot(Learning_curve);
                title('Learning Curve');
                xlabel('iteration'); ylabel('RMSE');
                grid on;
                
                o = complex2real(output, 2, nTarget) .* normal;
                t = complex2real(target, 2, nTarget) .* normal;
                temp_rmse = zeros(nTarget, 3);
                temp_mape = temp_rmse;
                temp_mae = temp_rmse;
                for i = 1:nTarget
                        %% Result
%                         disp(['● ', filename.output{i, 2}]);
                        figure('Name', ['exp', num2str(e), '_', filename.output{i, 2}, '_result']);
                        % original
                        data = dataset.output(:, i) .* normal(i);
                        plot(1:fs.nInput, data, 'b-');
                        hold on;
                        
                        % with output
                        ind = (1:nInput)+nFeatureDate;
                        result = data(ind)+o(:, i);
                        plot(ind+1, result, 'r--');
                        
                        % axis
                        x_min = 0;
                        x_max = nInput + nFeatureDate;
                        y_min = min([data;result]);
                        tmp = max([data;result]);
                        y_max = tmp * 25/24;
                        axis([x_min, x_max, y_min, y_max]);
                        
                        L = line([nTraining, nTraining]+nFeatureDate, [y_min, y_max]);
                        set(L, 'linestyle', '--', 'linewidth', 1.5, 'color', 'r');
                        
                        text(nTraining/2 + nFeatureDate, mean([tmp, y_max]), {'Training' ; 'phase'}, ...
                                'HorizontalAlignment', 'center', 'Color', 'r');
                        text((nTraining+nInput)/2 + nFeatureDate, mean([tmp, y_max]), {'Testing' ; 'phase'}, ...
                                'HorizontalAlignment', 'center', 'Color', 'r');
                        
                        title(filename.output{i, 2});
                        xlabel('Trading date index'); ylabel('Close price');
                        legend('Target', 'Forecast', 'location', 'southwest');
                        grid on; hold off;
                        
                        %% Error
                        figure('Name', ['exp', num2str(e), '_', filename.output{i, 2}, '_error']);
                        error = t(:, i) - o(:, i);
                        bar(ind+1, error);
                        title(filename.output{i, 2});
                        xlabel('Trading date index'); ylabel('Prediction error');
                        grid on;
                        
                        % axis
                        y_min = min(error);
                        tmp = max(error);
                        y_max = tmp * 4/3;
                        axis([x_min, x_max, y_min, y_max]);
                        
                        hold on;
                        L = line([nTraining, nTraining]+nFeatureDate, [y_min, y_max]);
                        set(L, 'linestyle', '--', 'linewidth', 1.5, 'color', 'r');
                        
                        text(nTraining/2 + nFeatureDate, mean([tmp, y_max]), {'Training' ; 'phase'}, ...
                                'HorizontalAlignment', 'center', 'Color', 'r');
                        text((nTraining+nInput)/2 + nFeatureDate, mean([tmp, y_max]), {'Testing' ; 'phase'}, ...
                                'HorizontalAlignment', 'center', 'Color', 'r');
                        
                        
                        %% Disp
                        temp_rmse(i, :) = [rmse(error(:)), rmse(error(1:nTraining)), rmse(error(nTraining+1:end))];
                        temp_mape(i, :) = [mape(data(ind+1), result), ...
                                mape(data(ind(1:nTraining)+1), result(1:nTraining)), ...
                                mape(data(ind(nTraining+1:end)+1), result(nTraining+1:end))];
                        temp_mae(i, :) = [mae(error(:)), mae(error(1:nTraining)), mae(error(nTraining+1:end))];
                        
                end
                error = t - o;
                exp_error{e} = [reshape(temp_rmse', 1, []); reshape(temp_mape', 1, []); reshape(temp_mae', 1, [])];
                exp_output{e} = o;
                
        end
        F = findall(0, 'type', 'figure');
        savefig(F, ['result_single_', filename.output{2}]);
        [~, a] = min(entire_RMSE);
        close(1:(nTarget*2+1)*(a-1)); close((nTarget*2+1)*a+1:(nTarget*2+1)*nExp);
        F = findall(0, 'type', 'figure');
        savefig(F, ['result_single_', filename.output{2}, '_thebest']);
        close all;
        save(['result_single_', filename.output{2}]);
        
end