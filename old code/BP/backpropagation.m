clc; clear; close all;
%% Function_sigmoid
sigmoid =@(x) 1./(1+exp(-x));
d_sigmoid =@(x) exp(-x)./(1+exp(-x)).^2;
%% Data pair
input = [[0, 0]; [0, 1]; [1, 0]; [1, 1]]';
target = [0, 1, 1, 0];
%% Weight
N_neural = [2, 1];
forward_neural = size(input, 1);     % input
for i = 1:length(N_neural)
        w{i} = rand(N_neural(i), forward_neural+1);
        forward_neural = N_neural(i);
end
%% training
N_train = 1000;
Learn_rate = 0.001;
N_data = size(input, 2);
for i = 1:N_train
        temp = [ones(1, size(input, 2)); input];
        for j = 1:length(N_neural)
                cal_input{j} = temp;
                temp = ones(1, size(input, 2));
                for k = 1:N_neural(j)
                        z{j}(k, :) = w{j}(k, :) * cal_input{j};
                        a{j}(k, :) = sigmoid(z{j}(k, :));
                        temp = [temp; a{j}(k, :)];
                end
        end
        output = temp(2:end, :);
        error = target - output;
        for j = length(N_neural):-1:1
                if j == length(N_neural)
                        delta = (-2/N_data.*error) .* d_sigmoid(z{j});    % dL/dy * dy/dz
                else
                        delta = for_delta .* d_sigmoid(z{j});    % dL/dy * dy/dz{2} * dz/da * da/dz{1}
                end
                delta_w =  delta * cal_input{j}';
                w{j} = w{j} - Learn_rate.*delta_w;
                if j ~= 1
                        for_delta = zeros(size(a{j-1}));
                        for k = 2:size(w{j}, 2)
                                for_delta = for_delta + delta .* w{j}(:, k)';
                        end
                end
        end
end