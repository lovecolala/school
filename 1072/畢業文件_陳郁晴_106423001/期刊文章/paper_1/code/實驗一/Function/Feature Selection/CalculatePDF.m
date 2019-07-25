function [y , pd, dx] = CalculatePDF(data)
% event: fitdist下的500個點
% pd_value: 500個點對應的Pd值
% pd: fitdist結果（kernal）
% dx: 500個點之間的距離

pd = fitdist(data, 'kernel');
x = linspace(pd.mean - 5*pd.std, pd.mean + 5*pd.std, 500);
dx = x(2) - x(1);
y = pdf(pd, x);

end