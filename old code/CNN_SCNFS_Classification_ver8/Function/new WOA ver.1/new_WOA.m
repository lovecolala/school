% Change log: 2018-10-29 vector version
% Change log: 2018-10-29 WOA-2 ver.4
%_________________________________________________________________________%
%  Whale Optimization Algorithm (WOA) source codes demo 1.0               %
%                                                                         %
%  Developed in MATLAB R2011b(7.13)                                       %
%                                                                         %
%  Author and programmer: Seyedali Mirjalili                              %
%                                                                         %
%         e-Mail: ali.mirjalili@gmail.com                                 %
%                 seyedali.mirjalili@griffithuni.edu.au                   %
%                                                                         %
%       Homepage: http://www.alimirjalili.com                             %
%                                                                         %
%   Main paper: S. Mirjalili, A. Lewis                                    %
%               The Whale Optimization Algorithm,                         %
%               Advances in Engineering Software , in press,              %
%               DOI: http://dx.doi.org/10.1016/j.advengsoft.2016.01.008   %
%                                                                         %
%_________________________________________________________________________%


% The Whale Optimization Algorithm
function [Leader_score,Leader_pos,Convergence_curve] = new_WOA(SearchAgents_no, Max_iter, Positions, dim, fobj, opts)

% initialize position vector and score for the leader
Leader_pos=zeros(1,dim);
Leader_score=inf; %change this to -inf for maximization problems

Convergence_curve=zeros(1,Max_iter);

t=0;% Loop counter

% Main loop
while t<Max_iter
        for i=1:size(Positions,1)
                % Calculate objective function for each search agent
                fitness=fobj(Positions(i,:), opts);
                
                % Update the leader
                if fitness<Leader_score % Change this to > for maximization problem
                        Leader_score=fitness; % Update alpha
                        Leader_pos=Positions(i,:); % 記錄迭代結束後最佳位置
                end
        end
        
        %     % 記錄標準差(std.)
        %     total_std(t+1,:) = std(Positions);
        %     % 記錄最佳位置變化
        %     total_best(t+1, :) = Leader_pos;
        %     % 記錄Cost變化
        %     total_score(t+1, :) = Leader_score;
        
        a=2-t*((2)/Max_iter); % a decreases linearly fron 2 to 0 in Eq. (2.3)
        a2=-1+t*((-1)/Max_iter); % a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        
        % Update the Position of search agents
        for i=1:size(Positions,1)
                pos_std = std(Positions); % std. of each dimensions. 每一個維度的標準差
                r1=rand(); % r1 is a random number in [0,1]
                r2=rand(); % r2 is a random number in [0,1]
                
                A=2*a*r1-a;  % Eq. (2.3) in the paper
                C=2*r2;      % Eq. (2.4) in the paper
                
                b=1;               %  parameters in Eq. (2.5)
                l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
                
                p = rand();        % p in Eq. (2.6)
                
                if p<1/3
                        if abs(A)>=1
                                % A的絕對值大於等於1，表示A的值可能是>=1或是<=1
                                % 就隨機選擇一隻鯨魚(agent)，迫使他離開原本與自己相近的鯨魚
                                rand_leader_index = randi(SearchAgents_no, 1, dim); % 在所有鯨魚(agent)裡面隨機挑中一隻
                                X_rand = Positions(([1:dim]-1).*SearchAgents_no + rand_leader_index); %Positions(rand_leader_index, :);
                                D_X_rand=abs(C*X_rand-Positions(i,:)); % Eq. (2.7)  % 計算這隻鯨魚和
                                Positions(i,:)=X_rand-A*D_X_rand;      % Eq. (2.8)  %
                        elseif abs(A)<1
                                % A<1, 表示現在位置已經進入迭代尾聲？
                                %Leader_neighbor = random_pos_ver3(Leader_pos, t, dim, pos_std);
                                Leader_neighbor = Leader_pos + randn() .* pos_std.*exp(-t) + t*exp(-t);
                                D_Leader_neighbor = C.*Leader_neighbor - Positions(i,:);
                                Positions(i,:) = Leader_pos + A.*D_Leader_neighbor;
                        end
                elseif p>=1/3 && p<2/3
                        %Leader_neighbor = random_pos_ver3(Leader_pos, t, dim, pos_std);
                        Leader_neighbor = Leader_pos + randn() .* pos_std.*exp(-t) + t*exp(-t);
                        distance2Leader = abs(Leader_neighbor - Positions(i,:));
                        Positions(i,:) = distance2Leader.*exp(b.*l).*cos(l.*2*pi) + Leader_pos;
                elseif p>=2/3
                        %Leader_neighbor = random_pos_ver3(Leader_pos, t, dim, pos_std);
                        Leader_neighbor = Leader_pos + randn() .* pos_std.*exp(-t) + t*exp(-t);
                        D_Leader=(10+rand*10).*(Leader_neighbor-Positions(i,:));
                        Positions(i,:)=Leader_pos + rand().*D_Leader;
                end
                
        end
        
        t=t+1;
        Convergence_curve(t)=Leader_score;
        % [t Leader_score]
end



