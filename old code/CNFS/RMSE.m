% ---Get Root Mean Square Error---
% e : array of errors
function y = RMSE(e)
    num_e = length(e);
    sum_e = sum(e.^2); 
y = (sum_e/num_e).^0.5;
