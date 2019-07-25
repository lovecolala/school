function [Leader_score, Leader_pos, Convergence_curve] = CACO(Max_iter, position, dim, fobj, antopts, opts)
% antopts.nAnt
% antopts.new_nAnt
% antopts.eva_rate
% antopts.learning_rate
% opts: for your model

%% Transfer option varible
nAnt = antopts.nAnt;
new_nAnt = antopts.new_nAnt;
eva_rate = antopts.eva_rate;
learning_rate = antopts.learning_rate;

Convergence_curve = zeros(1, Max_iter);

gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);


%% Training
for i = 1:size(position, 1)
        cost(i) = fobj(position, opts);
end
[cost, sort_index] = sort(cost);
position = position(sort_index, :);

for i = 1:Max_iter
        %% New ant and calculate cost
        weight = gauF(1:nAnt, 1, learning_rate * nAnt) ./ (learning_rate * nAnt * sqrt(2*pi));
        prob = weight ./ sum(weight);
        
        position = antrenew(position, nAnt, new_nAnt, dim, prob, eva_rate);
        for j = 1:size(position, 1)
                cost(j) = fobj(position(j, :), opts);
        end
        [cost, sort_index] = sort(cost);
        position = position(sort_index, :);
        Convergence_curve(i) = cost(1);
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
Leader_pos = position(1, :);
Leader_score = cost(1);

end