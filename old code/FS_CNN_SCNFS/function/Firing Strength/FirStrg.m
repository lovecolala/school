function beta = FirStrg(input, CenterSigma)
        % input =[h1; h2; h3]
        % center = [h1.core; h2.core; h3.core]
        % sigma = [h1.sig; h2.sig; h3.sig]
        center = CenterSigma(:,1);
        sigma = CenterSigma(:, 2);
        gauF=@(x) exp(-0.5.*(x-center).^2./sigma.^2);
        
        %% Calculate membership degree and beta
        MD = gauF(input);
        beta = prod(MD);
end