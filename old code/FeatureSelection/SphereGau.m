function beta = SphereGau(input, CenterSigma, nTarget)
center = CenterSigma(:,1);
sigma = CenterSigma(:, 2);

r = exp( -(input-center).^2 ./ (2*sigma.^2) );
theta = { -r .* (input-center) ./ sigma.^2      % dy/dh
        r .* ( ((input-center) ./ sigma) .^2 - 1 ) ./ sigma.^2};        %dy/d2h

u1 = r .* cos(theta{2}) .* cos(theta{1});
u2 = r .* cos(theta{2}) .* sin(theta{1});
u3 = r .* sin(theta{2});

j = sqrt(-1);
mu ={ u1 + j.*u2
        u1 + j.*u3
        u2 + j.*u3 };

for i = 1:ceil(nTarget/2)
        beta(i, :) = prod(mu{i}, 1);
end

end