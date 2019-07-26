function [swarm, cnn] = cnn_initial(cnn, opts)
nInput = 1;
nLayer = numel(cnn.layers);
mapsize = size(cnn.layers{1}.output{1});
mapsize = mapsize(1:end-1);
nSwarm = 1;

for L = 2:nLayer
        if strcmp(cnn.layers{L}.type, 'c')
                %% Convolution Layer
                t = [];
                mapsize = mapsize - cnn.layers{L}.filtersize + 1;
                
                for o = 1:cnn.layers{L}.nFilter
                        for i = 1:nInput
                                t = [t, nSwarm];
                                for k = 1:opts.nAnt
                                        swarm(nSwarm).ant(k).position = rand(cnn.layers{L}.filtersize) .* 2 - 1;
                                end
                                nSwarm = nSwarm + 1;
                        end
                end
                nInput = cnn.layers{L}.nFilter;
                cnn.layers{L}.swarm_index = t;
                
        elseif strcmp(cnn.layers{L}.type, 'p')
                %% Pooling Layer
                stride = cnn.layers{L}.stride;
                mapsize = ceil(mapsize ./ stride);
                
        end
end

end