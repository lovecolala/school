function [result, output] = model(whale, opts, rlse)
%% Options
cnn = opts.cnn;
premise_index = opts.premise_index;
sphere_index = opts.sphere_index;
C = opts.C;
nTraining = opts.nTraining;     % 185
nTarget = opts.nTarget;
nClass = opts.nClass;
target = opts.target;

%% Parameter
if nargin == 2
        rlse = 1;       % 要做RLSE（還在訓練階段）
else
        nTraining = size(target, 1);            % 不需要做RLSE（在做測試了，TD個數改變）
end
nRule = size(C, 1);
rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;

%% CNN
cnn = cnn_fp(cnn, whale);
CnnOutput = cnn_transfer(cnn);
nDim = size(CnnOutput, 1);     % 4
nInput = size(CnnOutput, 2);    % 205
A = repmat([ones(nInput, 1), CnnOutput'], 1, nRule);
A_train = A(1:nTraining, :);

%% Firing Strength
for j = 1:nRule
        Center = premise_index(j, :);
        CenterSigma = whale([Center', (Center+1)']);
        SCNFS_input = CnnOutput(:, 1:nTraining);
        beta(j, :, :) = SphereCom(SCNFS_input, CenterSigma, nTarget, nClass, whale(sphere_index));
end

%% Normalization
% beta(nRule, nTraining, nTarget*nClass)
beta = beta ./ sum(beta, 1);

%% RLSE
global THETA;
output = zeros(nTraining, nTarget*nClass);
if rlse == 1
        % Training
        THETA = zeros(nRule .* (nDim+1), nTarget*nClass);
        for i = 1:nTarget*nClass
                A_beta = A_train .* repelem(beta(:, :, i).', 1, 1+nDim);
                THETA(:, i) = RLSE(A_beta, target(:, i).', THETA(:, i));
                output(:, i) = A_beta * THETA(:, i);
        end
%         disp(THETA);
else
        % Testing
        for i = 1:nTarget*nClass
                A_beta = A .* repelem(beta(:, :, i).', 1, 1+nDim);
                output(:, i) = A_beta * THETA(:, i);
        end
end

%% RMSE
error = target(1:nTraining, :) - output;
RMSE = rmse(error(:));
result = RMSE;

end