function lambda = AOL(beta, aol, nThen)
nRule = size(beta, 1);
nInput = size(beta, 2);
nOutput = size(beta, 3);

lambda = zeros(nThen, nInput, nOutput);
for i  = 1:nOutput
        for q = 1:nRule
                lambda(:, :, i) = lambda(:, :, i) + cGauF(beta(q, :, i), aol(i).center, aol(i).sigma);
        end
end
lambda(isnan(lambda)) = 0;

end