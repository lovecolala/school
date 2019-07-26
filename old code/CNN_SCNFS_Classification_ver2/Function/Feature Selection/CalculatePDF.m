function [y , pd, dx] = CalculatePDF(data)
% event: fitdist�U��500���I
% pd_value: 500���I������Pd��
% pd: fitdist���G�]kernal�^
% dx: 500���I�������Z��

pd = fitdist(data, 'kernel');
x = linspace(pd.mean - 5*pd.std, pd.mean + 5*pd.std, 500);
dx = x(2) - x(1);
y = pdf(pd, x);

end