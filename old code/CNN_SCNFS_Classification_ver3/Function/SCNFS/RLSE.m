function [theta, P] = RLSE(A, y, theta)
% A = [a(1), a(2), a(3), ... , a(training)]
% and a is a vector of the THEN-part's parameter.

%% initial
nParameter = size(A, 2);
z = 10^9;
P = z * eye(nParameter);
nInput = size(A, 1);
%% calculate RLSE
for i = 1:nInput
        a = transpose(A(i, :));
        P = P - (P * a * transpose(a) * P) / (1 + transpose(a) * P * a);
        theta = theta + P * a * (y(i) - transpose(a) * theta);
end
end