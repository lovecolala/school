clc;
clear;
close all;

%% Get data pairs

x = linspace(-4,6,120);  % Get 120 points from x , range[-4,6]
y = (x-1).^2+1;          % function y

% parameter
num_data_pairs = 118;
num_training_data = 100;
num_inputs = 2;

% Computing
data_pairs = zeros(num_data_pairs,3);
for i = 1:num_data_pairs
    data_pairs(i,:) = [y(i) y(i+1) y(i+2)];
end

%% Clustering

cluster = clustering(data_pairs(1:num_training_data,:),num_inputs,0.3);

%% Construct Matrix

conMatrix = constructMatrix(num_inputs,[cluster(:).numOfCenter]);
CM_length = size(conMatrix,1);

%% Get firing strength

beta_sec = zeros(CM_length,1);
for i = 1:CM_length
    % firing strength
    beta = 1;
    for j = 1:num_inputs
        beta = beta.*cluster(j).GMF(conMatrix(i,j),:);
    end
    beta_sec(i,1) = sum(beta(:));
end
beta_avg = sum(beta_sec(:,1))/CM_length;

%% Set a new construct matrix after selecting

conMatrix_new = [];
beta_sec_new = [];
for i = 1:CM_length
    if beta_sec(i) > beta_avg
        conMatrix_new = [conMatrix_new; conMatrix(i,:)];
        beta_sec_new = [beta_sec_new; beta_sec(i)];
    end
end
CM_length_new = size(conMatrix_new,1);

%% Rule

% IF-part
class_index.num = 0;
class_index.set = [];

%Constructor
index = repmat(class_index, num_inputs, 1);
temp_start = 1;
temp_end = 0;
for i = 1:num_inputs
    u = unique(conMatrix_new(:,i));
    index(i).num = length(u);
    temp_end = temp_end + index(i).num;
    t = [temp_start:temp_end]';
    temp_start = temp_start + index(i).num;
    index(i).set = [u t];
end

%% PSO
%% Parameters of PSO

num_if_part = sum([index.num])*2;
num_then_part = CM_length_new*(num_inputs+1);

dimensionSize = [1 num_if_part+num_then_part];
num_Iteration = 500;
num_Particle = 50;

c0 = 0.85;
c1 = 2;
c2 = 2;
s1 = rand;
s2 = rand;


%% Initialization

% Parameters of Particle
class_Particle.pos = [];        
class_Particle.vel = [];
class_Particle.cost = [];
class_Particle.bestPos = [];
class_Particle.bestCost = [];

% Constructor
particle = repmat(class_Particle, num_Particle, 1); % size : num*1

globalBest.cost = inf ;

% Initialize
for i = 1:num_Particle
    % Set parameters of the particle
    particle(i).pos = rand(dimensionSize); 
    particle(i).vel = zeros(dimensionSize);
    particle(i).cost = costFunction_all_PSO(data_pairs(1:num_training_data,:), particle(i).pos, conMatrix_new, index, num_if_part);
    
    % Set Pbest
    particle(i).bestPos = particle(i).pos;
    particle(i).bestCost = particle(i).cost;
    
    % Set Gbest
    if particle(i).bestCost < globalBest.cost
        globalBest.pos = particle(i).pos;
        globalBest.cost = particle(i).cost;
    end
end

%% Main loop of PSO
rrmse = zeros(num_Iteration,1);
for t = 1:num_Iteration
    for i = 1:num_Particle
        
        % Update velocity and position of the particle
        particle(i).vel = c0*particle(i).vel ...
                        + c1*s1*(particle(i).bestPos - particle(i).pos) ...
                        + c2*s2*(globalBest.pos - particle(i).pos);
        particle(i).pos = particle(i).pos + particle(i).vel;
        
        % Get new cost
        particle(i).cost = costFunction_all_PSO(data_pairs(1:num_training_data,:), particle(i).pos, conMatrix_new, index, num_if_part);
        
        % Update the Pbest and Gbest
        if particle(i).cost < particle(i).bestCost
            particle(i).bestPos = particle(i).pos;
            particle(i).bestCost = particle(i).cost;
            
            if particle(i).bestCost < globalBest.cost
                globalBest.pos = particle(i).pos;
                globalBest.cost = particle(i).cost;
            end
        end  
    end
    rrmse(t) = real(globalBest.cost);
end

plot(x,y,'LineWidth',3);
hold on;
output = real(outputFunction_all_PSO(data_pairs, globalBest.pos, conMatrix_new, index, num_if_part));
plot(x(3:end),output,'ro');
figure;
semilogy(rrmse,'LineWidth',2);