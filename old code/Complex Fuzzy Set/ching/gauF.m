function MF = gauF(x, center, sigma)

MF = exp(-(x-center).^2./(2*sigma.^2));