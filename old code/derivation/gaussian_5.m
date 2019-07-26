%picture #1
x = linspace(0, 100);
c = 25;
sig = 3;
MF = exp(-(x-c).^2./(2*sig^2));
subplot(2, 3, 1); h = plot(x, MF);
grid on;
set(h, 'linewidth', 1.5, 'color', 'b');
title('c = 25, \sigma = 3');
xlabel('h');
ylabel('Membership degree');
%picture #2
c = 40;
MF = exp(-(x-c).^2./(2*sig^2));
subplot(2, 3, 2); h = plot(x, MF);
grid on;
set(h, 'linewidth', 1.5, 'color', 'g');
title('c = 40, \sigma = 3');
xlabel('h');
ylabel('Membership degree');
%picture #3
c = 50;
sig = 10;
MF = exp(-(x-c).^2./(2*sig^2));
subplot(2, 3, 3); h = plot(x, MF);
grid on;
set(h, 'linewidth', 1.5, 'color', 'y');
title('c = 50, \sigma = 10');
xlabel('h');
ylabel('Membership degree');
%picture #4
c = 25;
sig = 8;
MF = exp(-(x-c).^2./(2*sig^2));
subplot(2, 3, 4); h = plot(x, MF);
grid on;
set(h, 'linewidth', 1.5, 'color', 'c');
title('c = 25, \sigma = 8');
xlabel('h');
ylabel('Membership degree');
%picture #5
c = 40;
MF = exp(-(x-c).^2./(2*sig^2));
subplot(2, 3, 5); h = plot(x, MF);
grid on;
set(h, 'linewidth', 1.5, 'color', 'r');
title('c = 40, \sigma = 8');
xlabel('h');
ylabel('Membership degree');