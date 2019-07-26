function [next_theta, next_P] = RLSE(A, y, theta)
% A = [a(1), a(2), a(3), ... , a(training)]
% and a is a vector of the THEN-part's parameter.

%% initial
TD_size = size(A, 1);
Then_size = size(A, 2);
z = 10^9;
P = z * eye(Then_size);
if nargin == 2
        theta = zeros(Then_size, 1);
end
next_theta = theta;
%% calculate RLSE
for i = 1:TD_size
        a = transpose(A(i, :));
        P = P - (P * a * transpose(a) * P) / (1 + transpose(a) * P * a);
        next_theta = next_theta + P * a * (y(i) - transpose(a) * next_theta);
end
next_P = P;
end