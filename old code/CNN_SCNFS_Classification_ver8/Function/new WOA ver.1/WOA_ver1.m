function [Leader_score, Leader_pos, Convergence_curve] = WOA_ver1(SearchAgents_no, Max_iter, Positions, dim, fobj, opts)

Leader_pos = zeros(1, dim);
Leader_score = inf;

Convergence_curve = zeros(1, Max_iter);

%% Training
for t = 1:Max_iter
        for i=1:size(Positions,1)
                % Calculate objective function for each search agent
                fitness=fobj(Positions(i,:), opts);
                
                % Update the leader
                if fitness<Leader_score % Change this to > for maximization problem
                        Leader_score=fitness; % Update alpha
                        Leader_pos=Positions(i,:); % 記錄迭代結束後最佳位置
                end
        end
        
        a = (1-t/Max_iter)*2;          % a decreases linearly fron 2 to 0 in Eq. (2.3)
        a2 = -(1 + t/Max_iter);               % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        
        for i = 1:size(Positions,1)
                %% Encircling / Random phase parameter
                r1 = rand;            % r1 is a random number in [0,1]
                r2 = rand;            % r2 is a random number in [0,1]
                
                A = 2*a.*r1 - a;        % Eq. (2.3) in the paper
                C = 2.*r2;                % Eq. (2.4) in the paper
                
                %% Spiral phase parameter
                b = 1;               %  parameters in Eq. (2.5)
                l = (a2-1) * rand + 1;   %  parameters in Eq. (2.5)
                
                %% Updata position
                p = rand;        % p in Eq. (2.6)
                if p < 0.5
                        if abs(A) >=1
                                % random
                                % 每個維度找一個Positions
                                rand_leader_index = randi(SearchAgents_no, 1, dim);
                                X_rand = Positions(((1:dim) - 1) .* SearchAgents_no + rand_leader_index);          % 組一個random位置
                                D_X_rand = abs(C .* X_rand - Positions(i, :));
                                Positions(i, :) = X_rand - A .* D_X_rand;
                                
                        elseif abs(A) < 1
                                leader_index = abs(C .* Leader_pos - Positions(i, :));
                                Positions(i, :) = Leader_pos - A .* leader_index;
                        end
                elseif p>=0.5
                        sprial_index = abs(Leader_pos - Positions(i, :));
                        Positions(i, :) = Leader_pos + sprial_index .* exp(b.*l) .* cos(l.*2*pi);
                end
                
        end
        
        Convergence_curve(t) = Leader_score;
end
