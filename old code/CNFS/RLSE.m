%====== Recursive Least Square Estimator =====

function A = RLSE(p_dimensionSize,input,output)
%% Pararmeter Definition

len = size(input,1);
A = input;
y = output;
dimensionSize = [p_dimensionSize 1];

%% Parameters of RLSE

a = 10^9;
I = eye(p_dimensionSize);
P = a.*I;
shita = zeros(dimensionSize);

%% Main

for i=1:len
    P = P - P*A(i,:)'*A(i,:)*P / (1+A(i,:)*P*A(i,:)');
    shita = shita + P*A(i,:)'*(y(i)-A(i,:)*shita);
end

A = shita';