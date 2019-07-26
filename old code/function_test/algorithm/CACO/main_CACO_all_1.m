clc; clear; close all;
current_path = [pwd, '\'];
nDim = [500];
result_path = [current_path, 'algorithm\CACO\result\CACO_F38\'];

%% Option
nFunction = 38;
experience = 50;

opts.iteration = 1000;
opts.nAnt = 5;
opts.new_nAnt = 25;
opts.eva_rate = 0.9;
opts.learning_rate = 0.1;

function_name = cell(1, nFunction);
for i = 1:nFunction
        function_name{i} = ['F', int2str(i)];
end

for d = 1:length(nDim)
        if nDim(d) == 2
                nFunction = 26;
        end
        filename = ['CACO_', int2str(nDim(d)), '_1104'];
        
        %% result
        result_cost = zeros(experience, nFunction);
        result_time = zeros(experience, nFunction);
        learning_curve = cell(experience, nFunction);
        
        %% Training
        for f = 16:nFunction
                disp(['----- ', function_name{f}, ' -----']);
                %% Get Data
                [lb,ub, dim, func, bestSol] = Get_Functions_details(function_name{f});
                if dim ~= 100
                        opts.nDim = dim;
                else
                        opts.nDim = nDim(d);
                end
                
                if length(ub) == 1
                        opts.lb = repmat(lb, 1, opts.nDim);
                        opts.ub = repmat(ub, 1, opts.nDim);
                else
                        opts.lb = lb;
                        opts.ub = ub;
                end
                opts.func = func;
                
                %% Experience
                for e = 1:experience
                        %                 disp(['----- experience = ', int2str(e), ' -----']);
                        tic;
                        result = CACO(opts);
                        result_time(e, f) = toc;
                        learning_curve{e, f} = result.LearnCurve;
                        result_cost(e, f) = result.LearnCurve(end);
                end
                
        end
        
        save([result_path, filename], 'learning_curve', 'result_cost', 'result_time');
end