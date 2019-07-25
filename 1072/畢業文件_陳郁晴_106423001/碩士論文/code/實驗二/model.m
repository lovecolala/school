function [cost, output, theta] = model(position, opts, algo_rlse, train_rlse, theta)
%% Options
nn = opts.nn;
algoindex = opts.algoindex;
aol = opts.aol;
nThen = opts.nThen;
C = opts.C;
nTraining = opts.nTraining;
target = opts.target;
nTarget = opts.nTarget;

nOutput = ceil(nTarget/2);
[nDim, nInput] = size(nn.output);

%% Parameter
if nargin == 3
        train_rlse = 1;
end

if nargin <= 4
        if algo_rlse == 1
                theta = zeros(nThen .* (nDim+1), nOutput);
        else
                theta = position(algoindex.theta);
        end
end

if train_rlse == 0
        nTraining = size(target, 1);
end
nRule = size(C, 1);

%% NN
nn.weight = position(algoindex.nn.weight);
nn.bias = position(algoindex.nn.bias);
nn = nn_fp(nn);
A = repmat([ones(nInput, 1), nn.output'], 1, nThen);

%% SCNFS
for i = 1:nRule
        Center = algoindex.cs(i, :);
        CenterSigma = position([Center', (Center+1)']);
        scnfs.input = nn.output(:, 1:nTraining);
        if nOutput == 1
                beta(i, :) = SphereCom(scnfs.input, CenterSigma, nTarget, position(algoindex.lambda));
        else
                beta(i, :, :) = SphereCom(scnfs.input, CenterSigma, nTarget, position(algoindex.lambda));
        end
end
beta = beta ./ sum(beta, 1);
beta(isnan(beta)) = 0;

%% Aim Object
lambda = AOL(beta, aol, nThen);

%% RLSE
output = zeros(nTraining, nOutput);
for i = 1:nOutput
        A_beta = A(1:nTraining, :) .* repelem(lambda(:, :, i).', 1, nDim+1);
        if algo_rlse == 1 && train_rlse == 1
                theta(:, i) = RLSE(A_beta, target(:, i).', theta(:, i));
        end
        output(:, i) = A_beta * theta(:, i);
end

%% Performance index
error = complex2real(target(1:nTraining, :)-output, 2, nTarget);
cost = rms(error(:));

end