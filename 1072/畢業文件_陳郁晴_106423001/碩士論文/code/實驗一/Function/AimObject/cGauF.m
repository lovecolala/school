function result = cGauF(input, center, sigma)
if imag(input(1)) ~= 0 || imag(center(1)) ~= 0
        gauF =@(x) exp( -((x-center)' .' .* (x-center)) ./ (2 .* (sigma' .' .* sigma)) );
        dGauF =@(x) real(-gauF(x) .*( (x-center) ./ (sigma' .' .* sigma) ));
else
        gauF =@(x) exp(-0.5.*(x-center).^2./sigma.^2);
        dGauF =@(x) -gauF(x) .*( (x-center) ./ sigma.^2 );
end
r = gauF(input);
w = dGauF(input);
result = r.*exp(w .* sqrt(-1));

end