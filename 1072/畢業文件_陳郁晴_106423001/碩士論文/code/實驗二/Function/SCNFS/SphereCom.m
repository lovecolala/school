function beta = SphereCom(x, CenterSigma, nTarget, a)
center = CenterSigma(:, 1);
sigma = CenterSigma(:, 2);
% nU = nClass*nTarget;
nU = nTarget;
nTheta = 19;   % ←目前theta的個數
lambda = zeros(1, nTheta);
if nargin == 3
        a = ones(1, nU-1);
end
lambda(:, 1:nU-1) = a;

%% Gaussian (r) and theta (θ)
r = exp( -(x-center).^2 ./ (2*sigma.^2) );
xc = x-center;
xc_s = xc ./ sigma;
% 先這樣，之前要改再改
theta = { -r .* xc ./ sigma.^2 .* lambda(1)           % dy/dh
        r .* xc ./ sigma.^2 .* lambda(2)                     % dy/dc
        r .* xc.^2 ./ sigma.^3 .* lambda(3)               % dy/ds
        r .* ( xc_s .^2 - 1 ) ./ sigma.^2 .* lambda(4)                                        % dy/d2h
        r .* (xc_s.^2-1) ./ sigma.^2 .* lambda(5)                                             % dy/d2c
        r .* xc.^2 ./ sigma.^4 .* (xc_s.^2 - 3) .* lambda(6)                            % dy/d2s
        -r .* ( ((x-center) ./ sigma) .^2 - 1 ) ./ sigma.^2 .* lambda(7)          % dy/dhdc
        -r .* (x-center)./sigma.^3 .* (xc_s.^2-2) .* lambda(8)                       % dy/dhds
        r .* (x-center) ./ sigma.^3 .* (xc_s.^2 - 2) .* lambda(9)                     % dy/dcds
        r .* xc ./ sigma.^4 .* (-xc_s.^2 + 3) .* lambda(10)                                                       % dy/d3h
        r .* xc ./ sigma.^4 .* (xc_s.^2-3) .* lambda(11)                                                            % dy/d2hdc
        r ./ sigma.^3 .* (xc_s.^4 - 5.*xc_s.^2 + 2) .* lambda(12)                                           % dy/d2hds
        r .* xc ./ sigma.^4 .* (xc_s.^2 - 3) .* lambda(13)                                                          % dy/d3c
        r .* xc ./ sigma.^4 .* (3 - xc_s.^2) .* lambda(14)                                                          % dy/d2cdh
        r ./ sigma.^3 .* (xc_s.^4 - 5.*xc_s.^2 + 2) .* lambda(15)                                           % dy/d2cds
        r .* xc.^2 ./ sigma.^5 .* (xc_s.^4 - 9 .* xc_s.^2 + 12) .* lambda(16)                       % dy/d3s
        -r .* xc ./ sigma.^4 .* (xc_s.^4 - 7 .* xc_s.^2 + 6) .* lambda(17)                              % dy/d2sdh
        r .* xc ./ sigma.^4 .* (xc_s.^4 - 7 .* xc_s.^2 + 6) .* lambda(18)                                % dy/d2sdc
        -r ./ sigma.^3 .* (xc_s.^4 - 5 .* xc_s.^2 - 2) .* lambda(19)                                         % dy/dhdchs
        };

%% u1, u2, u3
u(:, :, 1) = r;
u = repmat(u, 1, 1, nU);
for i = nU:-1:2
        u(:, :, i) = u(:, :, i) .* sin(theta{i-1});
        u(:, :, 1:i-1) = u(:, :, 1:i-1) .* cos(theta{i-1});
end
% u = abs(u);
% u(u==0) = realmin;

%% membership degree
% mu(nCluster, nData, nTarget*nClass)
% mu = u;

        
for i = 1:size(u, 3)
        mu = u(:, :, 1:2:nU) + u(:, :, 2:2:nU) .* sqrt(-1);
        if mod(nU, 2) == 1
                mu(:, :, end+1) = u(:, :, nU);
        end
end


%% beta (β)
beta = prod(mu, 1);

end