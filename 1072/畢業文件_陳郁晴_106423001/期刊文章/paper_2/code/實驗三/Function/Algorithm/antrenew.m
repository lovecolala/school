function ant = antrenew(ant, nAnt, new_nAnt, dim, prob, eva_rate)
%% Calculate the sigma
sigma = zeros(nAnt, dim);
for i = 1:nAnt
        Dist = 0;
        for j = 1:nAnt
                Dist = Dist + abs(ant(i, :) - ant(j, :));
        end
        sigma(i, :) = eva_rate .* Dist/(nAnt-1);
end

%% Roulette Wheel Selection to new Ant
ant(new_nAnt+nAnt, dim) = 0;
for i = (1:new_nAnt) + nAnt
        p = RouletteWheel(prob);
        ant(i, :) = ant(p, :) + ant(p, :) .* randn(1, dim);
end

end