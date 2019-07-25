function nn = nn_fp(nn, transferfunc)
nn.output = nn.input * nn.weight + nn.bias;

if nargin == 2
        
        if strcmp(transferfunc, 'sigmoid')

                nn.output = sigmf(nn.output, [1, 0]);

        elseif strcmp(transferfunc, 'tanh')

                nn.output = tanh(nn.output);

        elseif strcmp(transferfunc, 'relu')

                nn.output = max(nn.output, 0);

        end
        
end


nn.output = nn.output';

end