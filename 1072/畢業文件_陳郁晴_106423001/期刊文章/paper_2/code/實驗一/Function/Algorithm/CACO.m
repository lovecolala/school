function [Leader_score, Leader_pos, Learning_curve] = CACO(iteration, position, fobj, algoopts, opts)
% fobj = model
% algoopts.nAnt
% algoopts.new_nAnt
% algoopts.eva_rate
% algoopts.learning_rate
% algoopts.rlse
% opts for models

%% Transfer option varible
nAnt = algoopts.nAnt;
new_nAnt = algoopts.new_nAnt;
eva_rate = algoopts.eva_rate;
learning_rate = algoopts.learning_rate;
nDim = length(position);

Learning_curve = zeros(1, iteration);
gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2); 

%% Initialize
ant = zeros(nAnt, nDim);
cost = zeros(1, nAnt);
for i = 1:nAnt
        if i == 1
                ant(i, :) = position;
        else
                ant(i, :) = randn(1, nDim);
        end
        cost(i) = fobj(ant(i, :), opts, algoopts.rlse);
end
[cost, sort_index] = sort(cost);
ant = ant(sort_index, :);

%% Training
for i = 1:iteration
        weight = gauF(1:nAnt, 1, learning_rate * nAnt) ./ (learning_rate * nAnt * sqrt(2*pi));
        prob = weight ./ sum(weight);
        ant = antrenew(ant, nAnt, new_nAnt, nDim, prob, eva_rate);
        for j = 1:nAnt+new_nAnt
                cost(j) = fobj(ant(j, :), opts, algoopts.rlse);
        end
        [cost, sort_index] = sort(cost);
        ant = ant(sort_index, :);
        
        if i == 1 || precost > cost(1)
                t = 0;
                precost = cost(1);
        else
                t = t + 1;
                if t > 5
                        learning_rate = learning_rate * 0.95;
                end
        end
        
end
Leader_pos = ant(1, :);
Leader_score = cost(1);

end