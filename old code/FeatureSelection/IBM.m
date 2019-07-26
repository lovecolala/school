clc; clear; close all;
%% Read and Transfer Data
filename = 'IBM.csv';
Close = 4;              % from data
StockData = csvread(filename, 1, Close);
StockData = StockData(:, 1);
Delta = StockData(2:end, :) - StockData(1:end-1, :);
%% Analysis Data
pd_k = fitdist(Delta, 'kernel');
mu = pd_k.mean;
sigma = pd_k.std;
x = linspace(mu-5*sigma, mu+5*sigma, 500);
y = pdf(pd_k, x);
plot(x, y, 'linewidth', 1.5);
grid on;
xlabel('Event'); ylabel('Probability Density');
title('Probability Density Distribution');
%% Entropy
phi = max(y)+realmin;
dx = x(2)-x(1);
Entropy =@(p) sum(p.*log10(phi./p).*dx);
ent = Entropy(y);
disp(ent);