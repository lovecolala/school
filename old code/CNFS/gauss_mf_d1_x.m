% Gaussian Membership Function 1st derivative
function y = gauss_mf_d1_x(x, parameter)
c = parameter(1);
sigma = parameter(2);
tmp = (x - c)/sigma;
org = exp(-tmp.^2/2); % Gaussian Membership Function
der1 = -tmp./sigma;
y = der1.*org;

