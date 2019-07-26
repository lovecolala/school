clc; clear; close all;
%% Cost Function
a = 1; b = 1;
x = linspace(a-5, a+5, 120);
y = (x-a).^2+b;
training = 100;     % # of training data
%% Data pair
input = [y(1 : length(y)-2); y(2 : length(y)-1)];       % 2*118
target = y(3 : length(y));
% input = [y(1 : length(y)-3); y(2 : length(y)-2); y(3 : length(y)-1)];
% target = y(4 : length(y));
%% Subtracting clustering (SC)
% core and sigma
for i = 1:size(input, 1)
        [h{i}.core, h{i}.sig] = subclust(input(i, 1:training)', 0.3);
        h{i}.sig = ones(size(h{i}.core)).*h{i}.sig;
end
%% Construct Matrix
for i = 1:size(input,1)
        if i == 1
                t = '1:length(h{i}.core)';
        else
                t = [ t, ', 1:length(h{i}.core)'];
        end
end
eval(['C = allcomb(', t, ');']);
%                  h1   h2
%Rule1 ┌   1        1 ┐
%Rule2  |    1        2  |
%Rule3 └   1        3 ┘
%% Membership Degree Accumulate
Rule_Accu = zeros(1, size(C, 1));
for i = 1:size(C, 1)            % Rule
        CenterSigma = [];
        for k = 1:size(C, 2)    % Dimension
                CenterSigma = [CenterSigma; h{k}.core(C(i, k)), h{k}.sig(C(i, k))];
        end
        beta = FirStrgi(input(:, 1:training), CenterSigma);
        Rule_Accu(i) = sum(beta);
end
%% threshold
avg = mean(Rule_Accu);
t = 1;
for i = 1:size(C, 1)            % Rule
        if Rule_Accu(i) > avg
                new_C(t, :) = C(i, :);
                t = t+1;
        end
end
%% IF-part
IFpart_para = [];
Rule_index.center = [];         % 放center在IF-part中的index
index_fore_start = 0;
for i = 1:size(new_C, 2)                % Dimension
        [use_FS, start, index] = unique(new_C(:, i));
        index = index.*2-1;                                                     % center's index
        index = index + index_fore_start;
        Rule_index.center = [Rule_index.center, index];
        index_fore_start = max(max(Rule_index.center))+1;
        temp = [h{i}.core(use_FS), h{i}.sig(use_FS)];
        temp = reshape(temp', 1, []);
        IFpart_para = [IFpart_para, temp];
end
%% initial partical
num = 50;
for i = 1:num
        particle(i).position = IFpart_para;
        particle(i).velocity = randn(size(particle(i).position));
        particle(i).p_min = inf;
end
g_min = inf;
%% renew velocity parameter
c0 = 0.85;
c1 = 2; c2 = c1;    % or rand*0.4+1.8
%% training
T_round = 500;     % iterative of training
N_Rule = size(new_C, 1);
N_h = size(new_C, 2);
% input_para = [ones(1, size(input, 2)); input];
input_para = repmat([ones(1,  size(input, 2)); input], size(new_C, 1), 1);
for i = 1:T_round
        for q = 1:num         %particle
                particle(q).beta = [];
                for k = 1:N_Rule             %Rule
                        % calculate beta
                        Center = Rule_index.center(k, 1:N_h);
                        CenterSigma = particle(q).position([Center', (Center+1)']);
                        particle(q).beta(k, :) = FirStrgi(input(:, 1:training), CenterSigma);
                end
                % Normalization
                particle(q).beta = particle(q).beta ./ sum(particle(q).beta, 1);     % get real
                particle(q).beta = [repmat(particle(q).beta(1, :), 3, 1); repmat(particle(q).beta(2, :), 3, 1)];
%                 particle(q).beta = reshape(repmat(particle(q).beta', N_h+1, 1), 1, []);
                % RLSE
                theta = RLSE(input_para(:, 1:training).*particle(q).beta, target(1:training));
                % RMSE
                particle(q).output = (input_para(:, 1:training).*particle(q).beta)' * theta;
                particle(q).RMSE = RMSE(target(1:training) - particle(q).output');
                % renew pbest and gbest
                if particle(q).RMSE < particle(q).p_min
                        particle(q).pbest = particle(q).position;
                        particle(q).p_min = particle(q).RMSE;
                end
                if particle(q).RMSE < g_min
                        gbest = particle(q).position;
                        g_min = particle(q).RMSE;
                end                
        end
        gRMSE(i) = g_min;
        for q = 1:num         %particle
                xi1 = rand*0.5+0.4; xi2 = rand*0.5+0.4;
                particle(q).velocity = c0*particle(q).velocity + c1*xi1*(particle(q).pbest-particle(q).position) + c2*xi2*(gbest-particle(q).position);
                particle(q).position = particle(q).position + particle(q).velocity;
        end
end
%% testing
beta = [];
for i = 1:N_Rule
        Center = Rule_index.center(k, 1:N_h);
        CenterSigma = gbest([Center', (Center+1)']);
        beta(k, :) = FirStrgi(input, CenterSigma);
end
% Normalization
beta = real(beta);
beta = sum(beta ./ sum(beta), 2);     % get real
beta = reshape(repmat(beta', N_h+1, 1), 1, []);
theta = RLSE(input_para, target);
output = (input_para.*beta')' * theta;
%% draw
figure(2);
plot(x, y, 'linewidth', 1.5);
hold on; title('Predicted Function'); grid on;
plot(x((N_h+1):length(x)), output , 'ro');
figure(3); 
title('error');
plot(x((N_h+1):length(x)), target-output'); 
grid on;
%% calculate RMSE in training and testing
Train_Test_RMSE = [RMSE(target(1:training)-output(1:training)'), RMSE(target(training+1:end)-output(training+1:end)')];
display(Train_Test_RMSE);