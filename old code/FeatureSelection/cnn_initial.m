function cnn = cnn_initial(cnn, opts, nLabel)
nInput = 1;
nLayer = numel(cnn.layers);
mapsize = size(cnn.layers{1}.output{1}(:, :, 1));
for L = 2:nLayer
        if strcmp(cnn.layers{L}.type, 'c')
                %% Covolution Layer
                mapsize = mapsize - cnn.layers{L}.filtersize + 1;
                
                for o = 1:cnn.layers{L}.nFilter
                        for i = 1:nInput
                                cnn.layers{L}.filter{i, o} = rand(cnn.layers{L}.filtersize) .*2 -1;
                        end
                        cnn.layers{L}.bias(o) = 0;
                end
                nInput = cnn.layers{L}.nFilter;
                
        elseif strcmp(cnn.layers{L}.type, 'p')
                %% Pooling Layer
                stride = cnn.layers{L}.stride;
                mapsize = ceil(mapsize ./ stride);
                
        end
end
if opts.fullconnect
        nOutput = prod(mapsize) * nInput;
        
        cnn.full.weight = (rand(nLabel, nOutput).*2-1) .* sqrt( 6 / (nOutput + nLabel) );
        cnn.full.bias = zeros(nLabel, 1);
end
end