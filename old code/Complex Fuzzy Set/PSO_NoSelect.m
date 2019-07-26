% clc; clear; close all;
%% Cost Function
num_data = 120;
training = 100;
%--------original curve--------------
x = linspace(-4, 6, num_data);
y = (x-1).^2+1;
%----------Example 1-----------------
% x = linspace(3, 7, num_data);
% y = 0.08*(1.2*(x-1).*cos(3*x)) + (x-(x-1).*cos(3*x)).*sin(x);
%----------Example 2-----------------
% x = linspace(-8, 12, num_data);
% y = ((x-2).*(2*x-1))./(1+x.^2);
%----------Example 3-----------------
% x = linspace(0, 1, num_data);
% y = 0.1 + 1.2*x + 2.8*sin(4*pi*x.^2);
%% Data pair
N_h = 2;
input = [];
for i = 1:N_h
        input = [input; y(i:length(y)-(N_h-i+1))];
end
target = y(N_h +1 : length(y));
%% Subtracting clustering (SC)
% center and sigmama
for i = 1:size(input, 1)
        [h{i}.center, h{i}.sigma] = subclust(input(i, 1:training)', 0.3);
        h{i}.sigma = ones(size(h{i}.center)).*h{i}.sigma;
end
%% Construct Matrix
t = []; comma = [];
for i = 1:size(input,1)
        t = [ t, comma, '1:length(h{i}.center)'];
        comma = ', ';
end
eval(['C = allcomb(', t, ');']);
N_Rule = size(C, 1);
%% construct index
% option 1, if para include THEN-part
[Rule_index, para] =ConIndex(C, 1, h);
%% initial partical
num = 50;
for i = 1:num
%         para = [IF_para, randn(1, numel(Rule_index.TS))];
        particle(i).position = para .* randn(size(para));
        particle(i).velocity = randn(size(para));
        particle(i).p_min = inf;
        particle(i).pbest = particle(i).position;
end
GlobalBest.g_min = inf;
GlobalBest.position = particle(1).position;
%% renew velocity parameter
c0 = 0.85;
c1 = 2; c2 = c1;    % or rand*0.4+1.8
xi1 = rand; xi2 = rand;
%% training
iteration = 500;     % iterative of training
input_para = [ones(1, size(input, 2)); input];
train_input_para = input_para(:, 1:training);
for i = 1:iteration
        for q = 1:num         %particle
                for k = 1:N_Rule             %Rule
                        Center = Rule_index.center(k, :);
                        CenterSigma = particle(q).position([Center', (Center+1)']);
                        % calculate firing strength and output
                        particle(q).beta(k, :) = FirStrgi(input(:, 1:training), CenterSigma);
                        particle(q).TS(k, :) = particle(q).position(Rule_index.TS(k, :)) * train_input_para;
                end
                % calculate error
                particle(q).beta = particle(q).beta ./ sum(particle(q).beta, 1);
                particle(q).output = sum(particle(q).TS .* particle(q).beta);
                particle(q).RMSE = RMSEi(target(1:training)-particle(q).output);
                % renew pbest and gbest
                if particle(q).RMSE < particle(q).p_min
                        particle(q).pbest = particle(q).position;
                        particle(q).p_min = particle(q).RMSE;
                end
                if particle(q).RMSE < GlobalBest.g_min
                        GlobalBest.position = particle(q).position;
                        GlobalBest.g_min = particle(q).RMSE;
                end                
                gRMSE(i) = GlobalBest.g_min;
                particle(q).velocity = c0*particle(q).velocity + c1*xi1*(particle(q).pbest-particle(q).position) + c2*xi2*(GlobalBest.position-particle(q).position);
                particle(q).position = particle(q).position + particle(q).velocity;
        end
end
%% testing
beta = zeros(size(C, 1), size(input, 2));
for i = 1:size(C, 1)
        Center = Rule_index.center(i, :);
        CenterSigma = GlobalBest.position([Center', (Center+1)']);
        beta(i, :) = FirStrgi(input, CenterSigma);
        TS(i, :) = GlobalBest.position(Rule_index.TS(i, :)) * input_para;
end
beta = beta ./ sum(beta, 1);         %Normalization
output = sum(TS .* beta);
%% draw
figure(2);
plot(x, y, 'linewidth', 1.5);
hold on; title('Predicted Function'); grid on;
plot(x((size(input, 1)+1):end), real(output) , 'ro');

figure(3); 
title('error');
plot(x((size(input, 1)+1):end), real(target-output), 'linewidth', 1.5); 
line([x(1), x(end)], [0, 0], 'color', 'k');
grid on;

figure(4);
plot(1:iteration, gRMSE, 'linewidth', 1.5);
title('RMSE of gbest'); xlabel('iteration'); ylabel('RMSE');
grid on;

figure(5);
semilogy(1:iteration, gRMSE, 'linewidth', 1.5);
title('RMSE of gbest'); xlabel('iteration'); ylabel('RMSE');
grid on;
%% calculate RMSE in training and testing
Train_Test_RMSE = [RMSEi(target(1:training)-output(1:training)), RMSEi(target(training+1:end)-output(training+1:end))];
All_RMSE = RMSEi(target - output);
display(Train_Test_RMSE);
display(All_RMSE);