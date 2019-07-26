function new_C  = BlockSelection(C, input, h, nTarget)
% beta_type = 'Sphere', use Sphere Complex Fuzzy Sets
% beta_type = 'Standard', use Standard Fuzzy Sets

nRule = size(C, 1);
nDim = size(C, 2);
%% ºâ¥X£]²Ö¥[­È (Rule_Accu)
Rule_Accu = zeros(1, nRule);
for i = 1:nRule
        CenterSigma = [];
        for k = 1:size(C, 2)    % Dimension
                CenterSigma = [CenterSigma; h{k}.center(C(i, k)), h{k}.sigma(C(i, k))];
        end
        beta = SphereCom(input, CenterSigma, nTarget);
        Rule_Accu(i) = sum(beta(:));
end
average = mean(Rule_Accu) + std(Rule_Accu);
%% New Construct Matrix
t = 1;
for i = 1:nRule
        if Rule_Accu(i) >= average
                new_C(t, :) = C(i, :);
                t = t+1;
        end
end

end