function FS = ComplexGau(input, center, sigma)
        % r(h)¡÷ gauF
        % £c(h)¡÷ £_gauF / £_h
        % ?(h) = r(h)*exp(j*£c(h))
        %         = Re(£g(h)+j*Im(£g(h))
        %         = r(h)*cos(£c(h)) + j*r(h)*sin(£c(h))
        % j = sqrt(-1);
        
        real_h = gauF(input, center, sigma);
        omage_h = real(-real_h .* ((input-center) ./ sigma.^2));
        image = exp(j .* omage_h);
        FS = real_h.*image;
        
%         FS = real .*cos(theta) + real .* sin(theta);
end