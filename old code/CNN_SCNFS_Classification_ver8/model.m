function [result, output] = model(whale, opts, rlse)
%% Options
cnn = opts.cnn;
index_rule = opts.index_rule;
index_sphere = opts.index_sphere;
C = opts.C;
nTraining = opts.nTraining;     % 185
target = opts.target;
nTarget = opts.nTarget;
nClass = opts.nClass;

nTargetClass = nClass*nTarget/2;

%% Parameter
if nargin == 2
        rlse = 1;       % 要做RLSE（還在訓練階段）
else
        nTraining = size(target.complex, 1);            % 不需要做RLSE（在做測試了，TD個數改變）
end
nRule = size(C, 1);
rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;

%% CNN
cnn = cnn_fp(cnn, whale);
nDim = size(cnn.output, 1);     % 4
nInput = size(cnn.output, 2);    % 205
A = repmat([ones(nInput, 1), cnn.output'], 1, nRule);
A_train = A(1:nTraining, :);

%% Firing Strength
for j = 1:nRule
        Center = index_rule(j, :);
        CenterSigma = whale([Center', (Center+1)']);
        SCNFS_input = cnn.output(:, 1:nTraining);
        beta(j, :, :) = SphereCom(SCNFS_input, CenterSigma, nTarget, nClass, whale(index_sphere));
end

%% Normalization
% beta(nRule, nTraining, nTarget*nClass)
beta = beta ./ sum(beta, 1);

%% RLSE
global THETA;
output.complex = zeros(nTraining, nTargetClass);
if rlse == 1
        % Training
        THETA = zeros(nRule .* (nDim+1), nTargetClass);
        for i = 1:nTargetClass
                A_beta = A_train .* repelem(beta(:, :, i).', 1, 1+nDim);
                THETA(:, i) = RLSE(A_beta, target.complex(:, i).', THETA(:, i));
                output.complex(:, i) = A_beta * THETA(:, i);
        end
else
        % Testing
        for i = 1:nTargetClass
                A_beta = A .* repelem(beta(:, :, i).', 1, 1+nDim);
                output.complex(:, i) = A_beta * THETA(:, i);
        end
end
output.real = complex2real(output.complex, 2);
output.real = sigmf(output.real, [10, 0]);

%% RMSE
error = target.real(1:nTraining, :) - output.real;
RMSE = rmse(error(:));

% 計算錯誤預測之筆數
% output = real2class(output, nClass, nTarget);
% diff = target.classification(1:nTraining, :) ~= output.classification;
% nError = sum(diff(:));
% [~, t] = max(target.real(1:nTraining, :), [], 2);
% [~, o] = max(output.real, [], 2);
% nError = sum(t ~= o);

% result = RMSE + nError;
result = RMSE;

end