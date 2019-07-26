function ant = antrenew(ant, opts)
%% Get Option value
nAnt = opts.nAnt;
new_nAnt = opts.new_nAnt;
nDim = opts.nDim;
prob = opts.prob;
eva_rate = opts.eva_rate;

%% Calculate the sigma
sigma = zeros(nAnt, nDim);
for i = 1:nAnt
        Dist = 0;
        for j = 1:nAnt
                Dist = Dist + abs(ant(i, :) - ant(j, :));
        end
        sigma(i, :) = eva_rate .* Dist/(nAnt-1);
end

%% Roulette Wheel Selection to new Ant
ant(new_nAnt+nAnt, nDim) = 0;
for i = (1:new_nAnt) + nAnt
        p = RouletteWheel(prob);
        ant(i, :) = ant(p, :) + ant(p, :) .* randn(1, nDim);
end

end