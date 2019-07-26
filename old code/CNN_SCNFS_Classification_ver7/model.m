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

if imag(target(1)) == 0
        nTargetClass = nTarget*nClass;
else
        nTargetClass = nTarget*nClass/2;
end

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
nDim = size(cnn.output, 1);     % 4
nInput = size(cnn.output, 2);    % 205
A = repmat([ones(nInput, 1), cnn.output'], 1, nRule);
A_train = A(1:nTraining, :);

%% Firing Strength
for j = 1:nRule
        Center = premise_index(j, :);
        CenterSigma = whale([Center', (Center+1)']);
        SCNFS_input = cnn.output(:, 1:nTraining);
        beta(j, :, :) = SphereCom(SCNFS_input, CenterSigma, nTarget, nClass, whale(sphere_index));
end

%% Normalization
% beta(nRule, nTraining, nTarget*nClass)
beta = beta ./ sum(beta, 1);

%% RLSE
global THETA;
output = zeros(nTraining, nTargetClass);
if rlse == 1
        % Training
        THETA = zeros(nRule .* (nDim+1), nTargetClass);
        for i = 1:nTargetClass
                A_beta = A_train .* repelem(beta(:, :, i).', 1, 1+nDim);
                THETA(:, i) = RLSE(A_beta, target(:, i).', THETA(:, i));
                output(:, i) = A_beta * THETA(:, i);
        end
%         disp(THETA);
else
        % Testing
        for i = 1:nTargetClass
                A_beta = A .* repelem(beta(:, :, i).', 1, 1+nDim);
                output(:, i) = A_beta * THETA(:, i);
        end
end

%% RMSE
error = target(1:nTraining, :) - output;
RMSE = rmse(error(:));
if imag(target(1)) ~= 0
        for i = 1:nTargetClass
                o(:, 2*i-1) = real(output(:, i));
                o(:, 2*i) = imag(output(:, i));
                t(:, 2*i-1) = real(target(1:nTraining, i));
                t(:, 2*i) = imag(target(1:nTraining, i));
        end
else
        o = output;
        t = target(1:nTraining, :);
end

for i = 1:nTarget
        index = (1:nClass)+(i-1)*nClass;
        [~, b] = max(o(:, index), [], 2);
        oc(:, i) = nClass - b;
        
        [~, b] = max(t(:, index), [], 2);
        tc(:, i) = nClass - b;
end
error_index = (tc ~= oc);
nError = sum(error_index(:));

result = RMSE + nError;

end