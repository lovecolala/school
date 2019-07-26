clc; close all; clear;
Entropy =@(p) p.*log10(1./p);
prop = linspace(0, 1);
% prop = rand(1, 10);
% prop = prop./sum(prop);
H = Entropy(prop);
plot(prop, H, '*');
axis([0, 1, 0, 0.2]);
xlabel("propability"); ylabel("Disorder (Chaos)");
title("Entropy Distribution");
grid on;
sum(prop)