function [Leader_score, Leader_pos, Learning_curve] = ABC(iteration, position, fobj, algoopts, opts)
% fobj = model
% algoopts.nBee
% algoopts.nOnlooker
% opts for models

%% Transfer option varible
nBee = algoopts.nBee;
nOnlooker = algoopts.nOnlooker;
limit = algoopts.limit;
nDim = length(position);

Learning_curve = zeros(1, iteration);
abandon = zeros(1, nBee);

%% Initialize
bee = zeros(nBee, nDim);
cost = zeros(1, nBee);
for i = 1:nBee
        if i == 1
                bee(i, :) = position;
        else
                bee(i, :) = position + randn(1, nDim);
        end
        cost(i) = fobj(bee(i, :), opts, algoopts.rlse);
end
[~, min_idx] = min(cost);
best.pos = bee(min_idx, :);
best.cost = cost(min_idx);

%% Training
for i = 1:iteration
        %% Search new position
        r = rand(nBee, 1) .* 2 - 1;
        k = randi(nBee-1, 1, nBee);
        k(k>=1:nBee) = k(k>=1:nBee) + 1;
        temp_pos = bee + r .* (bee - bee(k, :));
        for j = 1:nBee
                temp_cost = fobj(temp_pos(j, :), opts, algoopts.rlse);
                if temp_cost < cost(j) || isnan(cost(j))
                        bee(j, :) = temp_pos(j, :);
                        cost(j) = temp_cost;
                else
                        abandon(j) = abandon(j) + 1;
                end
        end
        
        %% Cost and probability 
        cost(isinf(cost)) = realmax;
        mean_cost = mean(cost);
        fitness = exp(-cost ./ mean_cost);
        prob = fitness ./ sum(fitness);
        r = rand(nOnlooker, 1) .* 2 - 1;
        for m = 1:nOnlooker
                p(m) = RouletteWheel(prob);
        end
        k = randi(nBee-1, 1, nOnlooker);
        k(k>=p) = k(k>=p) + 1;
        temp_pos = bee + r .* (bee - bee(k, :));
        for m = 1:nOnlooker
                temp_cost = fobj(temp_pos(m, :), opts, algoopts.rlse);
                if temp_cost < cost(p(m))
                        bee(p(m), :) = temp_pos(m, :);
                        cost(p(m)) = temp_cost;
                else
                        abandon(p(m)) = abandon(p(m)) + 1;
                end
        end
        
%         for j = 1:nBee
%                 if abandon(j) >= limit
%                         bee(j, :) =position + randn(1, nDim);
%                         cost(j) = fobj(bee(j, :), opts, algoopts.rlse);
%                         abandon(j) = 0;
%                 end
%         end
        [~, min_idx] = min(cost);
        best.pos = bee(min_idx, :);
        best.cost = cost(min_idx);
        
        Learning_curve(i) = best.cost;
        
end
Leader_score = best.cost;
Leader_pos = best.pos;

end