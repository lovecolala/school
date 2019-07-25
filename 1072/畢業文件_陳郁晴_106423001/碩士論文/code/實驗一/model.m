function [cost, output] = model(position, opts, rlse)
%% Options
nn = opts.nn;
antindex = opts.antindex;
aol = opts.aol;
nThen = opts.nThen;
C = opts.C;
nTraining = opts.nTraining;
target = opts.target;
nTarget = opts.nTarget;

nOutput = ceil(nTarget/2);

%% Parameter
if nargin == 2
        rlse = 1;
else
        nTraining = size(target, 1);
end
nRule = size(C, 1);

rmse =@(error) ( (error'*error) / length(error) ) ^ 0.5;

%% NN
nn.weight = position(antindex.nn.weight);
nn.bias = position(antindex.nn.bias);
nn = nn_fp(nn, nn.transferfunc);
nDim = size(nn.output, 1);
A = repmat([ones(size(nn.output, 2), 1), nn.output'], 1, nThen);
A_train = A(1:nTraining, :);

%% SCNFS
for i = 1:nRule
        Center = antindex.cs(i, :);
        CenterSigma = position([Center', (Center+1)']);
        scnfs.input = nn.output(:, 1:nTraining);
        if nOutput == 1
                beta(i, :) = SphereCom(scnfs.input, CenterSigma, nTarget, position(antindex.lambda));
        else
                beta(i, :, :) = SphereCom(scnfs.input, CenterSigma, nTarget, position(antindex.lambda));
        end
end

beta = beta ./ sum(beta, 1);

%% Aim Object
lambda = AOL(beta, aol, nThen);

%% RLSE
global THETA;
output = zeros(nTraining, nOutput);
if rlse == 1
        THETA = zeros(nThen .* (nDim+1), nOutput);
        for i = 1:nOutput
                A_beta = A_train .* repelem(lambda(:, :, i).', 1, nDim+1);
                THETA(:, i) = RLSE(A_beta, target(:, i).', THETA(:, i));
                output(:, i) = A_beta * THETA(:, i);
        end
else
        for i = 1:nOutput
                A_beta = A .* repelem(lambda(:, :, i).', 1, nDim+1);
                output(:, i) = A_beta * THETA(:, i);
        end
end

%% Performance index
error = complex2real(target(1:nTraining, :) - output, 2, nTarget);
% error = target(1:nTraining, :) - output;
cost = rmse(error(:));

end