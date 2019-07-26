function result = model(whale, opts, rlse)
%% Options
cnn = opts.cnn;
premise_index = opts.premise_index;
sphere_index = opts.sphere_index;
C = opts.C;
nTraining = opts.nTraining;
target = opts.target;

%% Parameter
if nargin == 2
        rlse = 1;
else
        nTraining = size(target, 1);
end
nRule = size(C, 1);
nTarget = size(target, 2);
rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;

%% CNN
cnn = cnn_fp(cnn, whale);
CnnOutput = cnn_transfer(cnn);
nDim = size(CnnOutput, 1);
nInput = size(CnnOutput, 2);
A = repmat([ones(nInput, 1), CnnOutput'], 1, nRule);
A_train = A(1:nTraining, :);

%% Theta
theta = zeros(nRule .* (nDim+1), nTarget);

%% Firing Strength
for j = 1:nRule
        Center = premise_index(j, :);
        CenterSigma = whale([Center', (Center+1)']);
        SCNFS_input = CnnOutput(:, 1:nTraining);
        beta(j, :, :) = SphereCom(SCNFS_input, CenterSigma, nTarget, whale(sphere_index));
end

%% Normalization
beta = beta ./ sum(beta, 1);

%% RLSE
if rlse == 1
        % Training
        for j = 1:nTarget
                A_beta = A_train .* repelem(beta(:, :, j).', 1, 1+nDim);
                theta(:, j) = RLSE(A_beta, target(:, j).', theta(:, j));
                output(:, j) = A_beta * theta(:, j);
        end
else
        % Testing
        for j = 1:nTarget
                A_beta = A .* repelem(beta(:, :, 1).', 1, 1+nDim);
                load('theta');
                output(:, j) = A_beta * theta(:, j);
        end
        save('output', 'output');
end

%% RMSE
error = target(1:nTraining, :) - output;
RMSE = rmse(error(:));
save('theta', 'theta');
result = RMSE;

end