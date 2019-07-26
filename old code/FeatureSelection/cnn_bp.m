function cnn = cnn_bp(cnn, target, opts)
nLayer = numel(cnn.layers);
alpha = opts.alpha;
dTransferFunc =@(output) (1+output) .* (1-output);

%% output layer
cnn.error = cnn.output - target;
cnn.mse = sum(cnn.error(:) .^2) ./ size(cnn.error, 2) ./ 2;

cnn.dz = cnn.error .* dTransferFunc(cnn.output);

%% Fully Connected Layer
cnn.full.dinput = cnn.full.weight' * cnn.dz;
if strcmp(cnn.layers{end}.type, 'c')
        t = cnn.full.input;
        cnn.full.dinput = cnn.full.dinput .* dTransferFunc(t);
end

sizeOutput = size(cnn.layers{end}.output{1});
nOutput = prod(sizeOutput(1:2));

for i = 1:numel(cnn.layers{end}.output)
        dz = cnn.full.dinput( ( (i-1) * nOutput+1) : i*nOutput, :);
        cnn.layers{end}.dz{i} = reshape( dz, sizeOutput(1), sizeOutput(2), sizeOutput(3) );
end

%% The Hidden Layer
for L = (nLayer-1):-1:1
        if strcmp(cnn.layers{L}.type, 'c')
                for i = 1:numel(cnn.layers{L}.output)
                        output = cnn.layers{L}.output{i};
                        backward_dz = cnn.layers{L+1}.dz{i};
                        stride = cnn.layers{L+1}.stride;
                        
                        cnn.layers{L}.da{i} = repelem(backward_dz, stride(1), stride(2), 1);
                        cnn.layers{L}.dz{i} = dTransferFunc(output) .* cnn.layers{L}.da{i};
                end
        elseif strcmp(cnn.layers{L}.type, 'p')
                for i = 1:numel(cnn.layers{L}.output)
                        z = zeros(size(cnn.layers{L}.output{1}));
                        for o = 1:numel(cnn.layers{L+1}.output)
                                backward_dz = cnn.layers{L+1}.dz{i};
                                filter = cnn.layers{L+1}.filter{i, o};
                                
                                z = z + convn(backward_dz, rot90(rot90(filter)), 'full');
                        end
                        cnn.layers{L}.dz{i} = z;
                end
        end
end

%% Update the filter
for L = 2:nLayer
        if strcmp(cnn.layers{L}.type, 'c')
                for o = 1:numel(cnn.layers{L}.output)
                        nData = size(cnn.layers{L}.dz{o}, 3);
                        output = cnn.layers{L}.dz{o};
                        for i = 1:numel(cnn.layers{L-1}.output)
                                input = cnn.layers{L-1}.output{i};
                                cnn.layers{L}.dfilter{i, o} = convn(input, output, 'valid') / nData;
                                cnn.layers{L}.filter{i, o} = cnn.layers{L}.filter{i, o} - alpha * cnn.layers{L}.dfilter{i, o};
                        end
                        cnn.layers{L}.dbias(o) = sum(output(:)) / nData;
                        cnn.layers{L}.bias(o) = cnn.layers{L}.bias(o) - alpha * cnn.layers{L}.dbias(o);
                end
        end
end
cnn.full.dweight = cnn.dz * cnn.full.input' / size(cnn.dz, 2);
cnn.full.dbias = mean(cnn.dz, 2);

cnn.full.weight = cnn.full.weight - alpha * cnn.full.dweight;
cnn.full.bias = cnn.full.bias - alpha * cnn.full.dbias;

end