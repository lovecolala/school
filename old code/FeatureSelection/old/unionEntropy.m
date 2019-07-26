function result = unionEntropy(pd1, F1, pd2, F2)

[xx, yy] = meshgrid(pd1, pd2);
pdxy = xx.*yy;
% phi = max(F1.phi, F2.phi);
dx = F1.dx;
dy = F2.dx;

result = - sum(sum(pdxy .* log10(pdxy))) .* dx .* dy;

end