% Gaussian Membership Function
function y = gauss_mf(x, parameter)
c = parameter(1);
sigma = parameter(2);
tmp = (x - c)/sigma;
y = exp(-tmp.^2/2);
