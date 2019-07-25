% 由於本篇有30天為feature
% 日期第一天的推算為文獻測試階段的第一天回推30天

clc; clear; close all;
date = {'17-Sep-2009', '31-Dec-2013'
        '30-Jan-2015', '3-Jul-2015'
        '19-Aug-2004', '11-Feb-2005'
        '17-Nov-2010', '31-Dec-2011'
        '16-Mar-2006', '1-May-2016'
        '17-Nov-2008', '1-Jan-2012'};
for dd = 6:size(date, 1)
        load('result_exp3');
        
        rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;
        mape =@(target, output) mean(abs((target-output)./target));
        mae =@(error) mean(abs(error));
        
        filename.input = {
                '^DJI'
                '^GSPC'
                '^IXIC'
                '^NYA'
                };
        filename.output ={
                'AAPL', 'Apple Inc.'
                'MSFT', 'Microsoft'
                'GOOG', 'Alphabet Inc.'
                'T', 'AT&T'
                };
        
        %% input and target
        dataset.input = [];
        input_table = cell(1, nTarget);
        target_table = cell(1, nTarget);
        for i = 1:nStock
                input_table{i} = getMarketDataViaYahoo(filename.input{i}, date{dd, 1}, date{dd, 2});
                t = input_table{i}{:, 2:5};
                t = t ./ max(abs(t));
                dataset.input = [dataset.input, t];
        end
        
        dataset.output =[];
        for i = 1:nTarget
                target_table{i} = getMarketDataViaYahoo(filename.output{i, 1}, date{dd, 1}, date{dd, 2});
                t = target_table{i}{:, 5};
                dataset.output = [dataset.output, t];
        end
        fs.input = dataset.input(2:end, :) - dataset.input(1:end-1, :);
        fs.target = dataset.output(2:end, :) - dataset.output(1:end-1, :);
        [fs.feature_table, fs.target_table] = construct_feature(fs.input, fs.target, nFeatureDate);
        fs.output = fs.feature_table(:, fs.FP);
        
        %% NN
        nn.input = fs.output;
        
        nn.weight = Leader_pos(antindex.nn.weight);
        nn.bias = Leader_pos(antindex.nn.bias);
        nn = nn_fp(nn, nn.transferfunc);
        nDim = size(nn.output, 1);
        A = repmat([ones(size(nn.output, 2), 1), nn.output'], 1, nThen);
        A_train = A;
        
        %% SCNFS
        beta = [];
        for i = 1:nRule
                Center = antindex.cs(i, :);
                CenterSigma = Leader_pos([Center', (Center+1)']);
                scnfs.input = nn.output;
                if nOutput == 1
                        beta(i, :) = SphereCom(scnfs.input, CenterSigma, nTarget, Leader_pos(antindex.lambda));
                else
                        beta(i, :, :) = SphereCom(scnfs.input, CenterSigma, nTarget, Leader_pos(antindex.lambda));
                end
        end
        beta = beta ./ sum(beta, 1);
        
        %% Aim Object
        lambda = AOL(beta, aol, nThen);
        
        %% RLSE
        output = [];
        for i = 1:nOutput
                A_beta = A .* repelem(lambda(:, :, i).', 1, nDim+1);
                output(:, i) = A_beta * THETA(:, i);
        end
        
        %% Calculate the result
        o = complex2real(output, 2, nTarget) .* normal;
        t = fs.target_table;
        t(t==0) = realmin;
        error = t - o;
        data = dataset.output(nFeatureDate+2:end, :);
        result = dataset.output(nFeatureDate+1:end-1, :)+o;
        result_error = [rms(error)
                mape(data, result)
                mae(error)];
        
        %% Save and figure
        save(['result_date', num2str(dd)]);
        for i = 1:nTarget
                figure('Name', ['date', num2str(dd), '_', filename.output{i, 2}, '_result']);
                date = target_table{i}{nFeatureDate+2:end, 1};
                plot(date, data(:, i), 'b-');
                hold on;
                plot(date, result(:, i), 'r--');
                title(filename.output{i, 2});
                xlabel('Trading date'); ylabel('Close price');
                legend('Target', 'Forecast', 'location', 'southeast');
                grid on; hold off;
                axis tight;
                
                figure('Name', ['date', num2str(dd), '_', filename.output{i, 2}, '_error']);
                plot(date, error(:, i));
                title(filename.output{i, 2});
                xlabel('Trading date'); ylabel('Prediction error');
                grid on;
                axis tight;
        end
        F = findall(0, 'type', 'figure');
        savefig(F, ['result_date', num2str(dd)]);
        close all;
        
end