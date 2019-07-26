syms x c sig
x = linespace(0, 100);
c = 50;
sig = 10;
f = exp(-(x-c).^2./(2*sig^2));
plot(x, diff(f, x, 1));