function [Leader_score, Leader_pos, Learning_curve] = WOA(iteration, position, fobj, algoopts, opts)
% fobj = model
% algoopts.nAgent
% opts for models

%% Transfer option varible
nAgent = algoopts.nAgent;
nDim = length(position);

Learning_curve = zeros(1, iteration);

%% Initialize
whale = zeros(nAgent, nDim);
cost = zeros(1, nAgent);
for i = 1:nAgent
        if i == 1
                whale(i, :) = position;
        else
                whale(i, :) = randn(1, nDim);
        end
        cost(i) = fobj(whale(i, :), opts, algoopts.rlse);
end
[~, min_idx] = min(cost);
best.pos = whale(min_idx, :);
best.cost = cost(min_idx);

%% Training
for i = 1:iteration
        a = (1-i/iteration)*2;
        a2 = -(1 + i/iteration);
        for j = 1:nAgent
                r1 = rand;
                r2 = rand;
                A = 2*a.*r1 - a;
                C = 2.*r2;
                
                b = 1;
                l = (a2-1) * rand + 1;
                
                p = rand;
                if p >= 0.5
                        dist_s = abs(best.pos - whale(j, :));
                        whale(j, :) = dist_s .* exp(b*l) .* cos(2*pi*l) + best.pos;
                        
                elseif abs(A) >= 1
                        index = randi(nAgent, 1, nDim);
                        index = ((1:nDim) - 1) .* nAgent + index;
                        X_rand = whale(index);
                        dist_rand = abs(C .* X_rand - whale(j, :));
                        whale(j, :) = X_rand - A .* dist_rand;
                        
                else
                        dist_e = abs(C .* best.pos - whale(j, :));
                        whale(j, :) = best.pos - A .* dist_e;
                        
                end
                cost = fobj(whale(j, :), opts, algoopts.rlse);
                if cost < best.cost
                        best.cost = cost;
                        best.pos = whale(j, :);
                end
                
        end
        Learning_curve(i) = best.cost;
end
Leader_score = best.cost;
Leader_pos = best.pos;

end