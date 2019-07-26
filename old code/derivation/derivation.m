c = 0;
sig = 2;
x = linspace(-10, 10);
MF = exp(-(x-c).^2./(2*sig^2));
h = plot(x, MF);
hold on;
grid on;
set(h, 'linewidth', 1.5);
xlabel('h');
ylabel('Membership Function');
MFdx = MF.*(-1).*(x-c)./sig^2;
h = plot(x, MFdx);
set(h, 'linewidth', 1.5);
MFdc = MF.*(x-c)./sig^2;
h = plot(x, MFdc);
set(h, 'linewidth', 1.5);
MFdsig = MF.*(x-c).^2./sig^3;
h = plot(x, MFdsig);
set(h, 'linewidth', 1.5);
title('one-derivative');
h = legend('y', 'dy/dx', 'dy/dc', 'dy/d\sigma');
set(h, 'fontsize', 15);