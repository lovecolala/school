function new_C  = BlockSelection(C, input, h, nTarget, nRule_max)
% beta_type = 'Sphere', use Sphere Complex Fuzzy Sets
% beta_type = 'Standard', use Standard Fuzzy Sets

nRule = size(C, 1);

if nargin == 4
        nRule_max = inf;
end

%% ��X�]�֥[�� (Rule_Accu)
Rule_Accu = zeros(1, nRule);
for i = 1:nRule
        CenterSigma = [];
        for k = 1:size(C, 2)    % Dimension
                CenterSigma = [CenterSigma; h{k}.center(C(i, k)), h{k}.sigma(C(i, k))];
        end
        beta = SphereCom(input, CenterSigma, nTarget);
        Rule_Accu(i) = sum(beta(:));
end
average = mean(Rule_Accu);

%% New Construct Matrix
ntemp = sum(Rule_Accu >= average);
nRule = min(nRule_max, ntemp);
[~, index] = sort(Rule_Accu, 'descend');
index = sort(index(1:nRule));
new_C = C(index, :);

end