function beta = FirStrgi(input, CenterSigma)
        % input =[h1; h2; h3]
        % center = [h1.core; h2.core; h3.core]
        % sigma = [h1.sig; h2.sig; h3.sig]
        center = CenterSigma(:,1);
        sigma = CenterSigma(:, 2);
        MD = ComplexGau(input, center, sigma);
        beta = prod(MD);        %¬Û­¼¥Î
end