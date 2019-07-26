function beta = SphereCom(x, CenterSigma, nTarget)
center = CenterSigma(:, 1);
sigma = CenterSigma(:, 2);

%% Gaussian (r) and theta (£c)
r = exp( -(x-center).^2 ./ (2*sigma.^2) );
theta = { -r .* (x-center) ./ sigma.^2      % dy/dh
        r .* ( ((x-center) ./ sigma) .^2 - 1 ) ./ sigma.^2};        %dy/d2h

%% u1, u2, u3
u1 = r .* cos(theta{2}) .* cos(theta{1});
u2 = r .* cos(theta{2}) .* sin(theta{1});
u3 = r .* sin(theta{2});

%% membership degree
j = sqrt(-1);
mu ={ u1 + j.*u2
        u1 + j.*u3
        u2 + j.*u3 };

%% beta (£])
for i = 1:ceil(nTarget/2)
        beta(1, :, i) = prod(mu{i});
end

end