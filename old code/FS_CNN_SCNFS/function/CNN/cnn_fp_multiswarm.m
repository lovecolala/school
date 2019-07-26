function cnn = cnn_fp_multiswarm(cnn, swarm, index)
nLayer = numel(cnn.layers);
nInput = 1;
s = 0;
transferfun=@(x) tanh(x);

for L = 2:nLayer
        if strcmp(cnn.layers{L}.type, 'c')
                %% Convolution Layer
                t = [cnn.layers{L}.filtersize - 1, 0];
                
                nFilter = cnn.layers{L}.nFilter;
                for o = 1:nFilter
                        z = zeros(size(cnn.layers{L-1}.output{1}) - t);
                        for i = 1:nInput
                                s = s + 1;
                                z = z + convn(cnn.layers{L-1}.output{i}, swarm(s).ant(index(s)).position, 'valid');
                        end
                        cnn.layers{L}.output{o} = transferfun(z);
                end
                
                nInput = cnn.layers{L}.nFilter;
                
        elseif strcmp(cnn.layers{L}.type, 'p')
                %% Pooling Layer
                for i = 1:nInput
                        input = cnn.layers{L-1}.output{i};
                        stride = cnn.layers{L}.stride;
                        % input(stride(1) * ceil( size(input, 1) / stride(1) ), stride(2) * ceil( size(input, 2) / stride(2) )) = 0;
                        
                        z = convn(input, ones(stride)./prod(stride), 'valid');
                        if ndims(z) == 3
                                cnn.layers{L}.output{i} = z(1:stride(1):end, 1:stride(2):end, :);
                        elseif ndims(z) == 4
                                cnn.layers{L}.output{i} = z(1:stride(1):end, 1:stride(2):end, 1:stride(3):end, :);
                        end
                end
                
        end
end

end