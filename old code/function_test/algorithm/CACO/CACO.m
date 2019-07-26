function result = CACO(opts)
if nargin == 0
        opts.iteration = 100;
        opts.nDim = 10;
        opts.nAnt = 10;
        opts.new_nAnt = 20;
        opts.eva_rate = 0.9;
        opts.learning_rate = 0.1;
        opts.lb = -ones(1, opts.nDim);
        opts.ub = ones(1, opts.nDim);
        opts.func =@(x) x;
        return;
end

%% Transfer option varible
iteration = opts.iteration;
nAnt = opts.nAnt;
new_nAnt = opts.new_nAnt;
all_nAnt = nAnt + new_nAnt;
nDim = opts.nDim;
learning_rate = opts.learning_rate;
lb = opts.lb;
ub = opts.ub;
func = opts.func;

gauF =@(x, center, sigma) exp(-0.5.*(x-center).^2./sigma.^2);
weight = gauF(1:nAnt, 1, learning_rate * nAnt) ./ (learning_rate * nAnt * sqrt(2*pi));
opts.prob = weight ./ sum(weight);

%% Initial
persistent ant;
ant = rand(nAnt, nDim) .* (ub-lb) + lb;
cost = func(ant);
[~, sort_index] = sort(cost);
ant = ant(sort_index, :);

%% Training
result.LearnCurve = zeros(1, iteration);
for i = 1:iteration
        %% New ant and calculate cost
        ant = antrenew(ant, opts);
        for j = (1:new_nAnt)+nAnt
                ant(j, :) = max([min([ant(j, :); ub]); lb]);
        end
        cost = func(ant);
        [cost, sort_index] = sort(cost);
        ant = ant(sort_index, :);
        result.LearnCurve(i) = cost(1);
        
end
result.position = ant(1, :);
end