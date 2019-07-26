c = 50;
sig = 10;
x = linspace(0, 100);
MF = exp(-(x-c).^2./(2*sig^2));
h = plot(x, MF);        %y
hold on;
grid on;
set(h, 'linewidth', 1.5);
xlabel('h');
ylabel('Membership Function');
MFdx = MF.*(-1).*(x-c)./sig^2;
h = plot(x, MFdx);      %dy/dx
set(h, 'linewidth', 1.5);
MFdc = MF.*(x-c)./sig^2;
h = plot(x, MFdc);      %dy/dc
set(h, 'linewidth', 1.5);
MFdsig = MF.*(x-c).^2./sig^3;
h = plot(x, MFdsig);    %dy/dsig
set(h, 'linewidth', 1.5);
MFdx2=MF./sig^2.*(((x-c)./sig).^2-1);
h = plot(x, MFdx2);     %dy/dx2
set(h, 'linewidth', 1.5);
MFdxdc=MF.*(-1)./sig^2.*(((x-c)./sig).^2-1);
h = plot(x, MFdxdc);    %dy/dxdc
set(h, 'linewidth', 1.5);
MFdxdsig = MF.*(-1).*(x-c)./sig^3.*(((x-c)./sig).^2-2);
h = plot(x, MFdxdsig);  %dy/dxdsig
set(h, 'linewidth', 1.5);
MFdc2 = MF./sig^2.*(((x-c)./sig).^2-1);
h = plot(x, MFdc2);     %dy/dc2
set(h, 'linewidth', 1.5);
MFdcdsig = MF.*(x-c)./sig^3.*(((x-c)./sig).^2-2);
h = plot(x, MFdcdsig);  %dy/dcdsig
set(h, 'linewidth', 1.5);
MFdsig2 = MF.*(x-c).^2./sig^4.*(((x-c)./sig).^2-3);
h = plot(x, MFdsig2);   %dy/dsig2
set(h, 'linewidth', 1.5);
legend('y', 'dy/dx', 'dy/dc', 'dy/d\sigma', 'dy/d^{2}x', 'dy/dxdc', 'dy/dxd\sigma', ...
    'dy/d^{2}c', 'dy/dcd\sigma', 'dy/d^{2}\sigma');