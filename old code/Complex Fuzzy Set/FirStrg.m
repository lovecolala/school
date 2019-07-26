function beta = FirStrg(input, CenterSigma)
        % input =[h1; h2; h3]
        % center = [h1.core; h2.core; h3.core]
        % sigma = [h1.sig; h2.sig; h3.sig]
        center = CenterSigma(:,1);
        sigma = CenterSigma(:, 2);
        MD = gauF(input, center, sigma);
        beta = prod(MD);
end