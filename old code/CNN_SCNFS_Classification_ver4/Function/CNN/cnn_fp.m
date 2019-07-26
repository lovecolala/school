function cnn = cnn_fp(cnn, whale)
nLayer = numel(cnn.layers);
nInput = 1;
% s = 0;
transferfun=@(x) x;

for L = 2:nLayer
        if strcmp(cnn.layers{L}.type, 'c')
                %% Convolution Layer
                t = [cnn.layers{L}.filtersize - 1, 0];
                
                nFilter = cnn.layers{L}.nFilter;
                for o = 1:nFilter
                        z = zeros(size(cnn.layers{L-1}.output{1}) - t);
                        for i = 1:nInput
                                z = z + convn(cnn.layers{L-1}.output{i}, whale(cnn.layers{L}.filter_index{i, o}), 'valid');
                        end
                        cnn.layers{L}.output{o} = transferfun(z);
                end
                nInput = cnn.layers{L}.nFilter;
                
        elseif strcmp(cnn.layers{L}.type, 'p')
                %% Pooling Layer
                for i = 1:nInput
                        input = cnn.layers{L-1}.output{i};
                        stride = cnn.layers{L}.stride;
                        z = convn(input, ones(stride)./prod(stride), 'valid');
                        cnn.layers{L}.output{i} = z(1:stride(1):end, 1:stride(2):end, :);
                        
                end
                
        end
end

end