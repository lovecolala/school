function cnn = cnn_initial(cnn)
nInput = 1;
nLayer = numel(cnn.layers);
mapsize = size(cnn.layers{1}.output{1});
mapsize = mapsize(1:end-1);

fore_index = 0;
for L = 2:nLayer
        if strcmp(cnn.layers{L}.type, 'c')
                %% Convolution Layer
                t = [];
                mapsize = mapsize - cnn.layers{L}.filtersize + 1;
                for o = 1:cnn.layers{L}.nFilter
                        for i = 1:nInput
                                filtersize = cnn.layers{L}.filtersize;
                                cnn.layers{L}.filter_index{i, o} = reshape(1:prod(filtersize), filtersize(2), filtersize(1))' + fore_index;
                                fore_index = fore_index + prod(filtersize);
                        end
                end
                nInput = cnn.layers{L}.nFilter;
                
        elseif strcmp(cnn.layers{L}.type, 'p')
                %% Pooling Layer
                stride = cnn.layers{L}.stride;
                mapsize = ceil(mapsize ./ stride);
                
        end
end

end