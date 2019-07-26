clc; clear; close all;
%% Get Data
function_name = 'F1';
[lb,ub, dim, func, bestSol] = Get_Functions_details(function_name);
opts.nDim = dim;

%% Option
opts.iteration = 500;
opts.nAnt = 10;
opts.new_nAnt = 90;
opts.eva_rate = 0.9;
opts.learning_rate = 0.1;
opts.lb = repmat(lb, 1, opts.nDim);
opts.ub = repmat(ub, 1, opts.nDim);
opts.func = func;

%% Training
tic
result = CACO(opts);
disp(toc);

%% Drawing
figure;
subplot(1, 2, 1);
plot(result.LearnCurve);
grid on;
title(function_name);
subplot(1, 2, 2);
semilogy(result.LearnCurve);
grid on;
title(function_name);

disp(['result = [', num2str(result.position), ']']);
disp(['best cost = ', num2str(result.LearnCurve(end))]);