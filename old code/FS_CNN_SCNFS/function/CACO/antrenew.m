function swarm = antrenew(swarm, opts)

nSwarm = numel(swarm);
for i = 1:nSwarm
        %% Calculate the sigma
        for j = 1:opts.nAnt
                Dist = 0;
                for k = 1:opts.nAnt
                        Dist = Dist + abs( swarm(i).ant(j).position - swarm(i).ant(k).position );
                end
                swarm(i).ant(j).sigma = opts.eva_rate .* Dist / (opts.nAnt-1);
        end
        
        %% Roulette Wheel Selection to new Ant
        size_ant = size(swarm(i).ant(1).position);
        for j = (1:opts.new_nAnt) + opts.nAnt
                swarm(i).ant(j).position = zeros(size_ant);
        end
        
        for j = (1:opts.new_nAnt) + opts.nAnt
                p =RouletteWheel(opts.prob);
                swarm(i).ant(j).position = swarm(i).ant(p).position + swarm(i).ant(p).sigma .* randn(size_ant);
        end
end

end