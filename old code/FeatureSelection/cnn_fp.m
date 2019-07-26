function cnn = cnn_fp(cnn, opts)
TransferFunc =@(z) tanh(z);
nLayer = numel(cnn.layers);
nInput = 1;
for L = 2:nLayer
        if strcmp(cnn.layers{L}.type, 'c')
                %% Convolution layer
                % 卷積層的輸出矩陣大小為Input大小-Filter大小+1
                t = [cnn.layers{L}.filtersize - 1, 0];
                
                for o = 1:cnn.layers{L}.nFilter
                        z = zeros(size(cnn.layers{L-1}.output{1}) - t);
                        for i = 1:nInput
                                z = z + convn(cnn.layers{L-1}.output{i}, cnn.layers{L}.filter{i, o}, 'valid');
                        end
                        z = z + cnn.layers{L}.bias(o);
                        cnn.layers{L}.output{o} = TransferFunc(z);
                end
                
                nInput = cnn.layers{L}.nFilter;
        elseif strcmp(cnn.layers{L}.type, 'p')
                %% Pooling layer
                for i = 1:nInput
                        input = cnn.layers{L-1}.output{i};
                        stride = cnn.layers{L}.stride;
                        input(stride(1) * ceil(size(input, 1)/stride(1)), stride(2) * ceil(size(input, 2)/stride(2))) = 0;
                        
                        z = convn(input, ones(stride) ./ prod(stride), 'valid');
                        cnn.layers{L}.output{i} = z(1:stride(1):end, 1:stride(2):end, :);
                end
        end
end

if opts.fullconnect
        cnn.full.input = [];
        for i = 1:numel(cnn.layers{end}.output)
                sizeOutput = size(cnn.layers{end}.output{i});
                t = reshape(cnn.layers{end}.output{i}, prod(sizeOutput(1:2)), sizeOutput(3));
                cnn.full.input = [cnn.full.input; t];
        end
        cnn.full.netinput = cnn.full.weight * cnn.full.input + cnn.full.bias;
        cnn.output = TransferFunc(cnn.full.netinput);
end

end