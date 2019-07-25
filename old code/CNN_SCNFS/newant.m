function ant = newant(ant, opts)
%% Calculate the sigma
for i = 1:opts.nAnt
        Dist = 0;
        for j = 1:opts.nAnt
                Dist = Dist + abs( ant(i).position - ant(j).position );
        end
        ant(i).sigma = opts.eva_rate .* Dist / (opts.nAnt-1);
end

%% Roulette Wheel Selection to new Ant
size_ant = size(ant(1).position);
for i = (1:opts.new_nAnt) + opts.nAnt
        ant(i).position = zeros(size_ant);
end

for i = (1:opts.new_nAnt) + opts.nAnt
        for j = 1:numel(ant(1).position)
                p =RouletteWheel(opts.prob);
                ant(i).position(j) = ant(p).position(j) + ant(p).sigma(j)*randn;
        end
end
end