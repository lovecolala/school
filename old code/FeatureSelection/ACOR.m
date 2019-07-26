function [Ant, opts, learncurve] = ACOR(cnn, Ant, prob, x, y, opts)
% 第一次進來時Ant只有一個
learncurve = [];
nAnt = opts.nAnt;
iteration = opts.iteration;
kxi = opts.kxi;
nLayer = numel(Ant{1}.layers);
alpha = opts.antalpha;
nNewAnt = opts.nNewAnt;

cost = zeros(1, nAnt+nNewAnt);

for i = 1:iteration
        disp(['iteration: ', num2str(i), ' / ', num2str(iteration)]);
        %% calculate sigma
        antsigma = [];
        antsigma.layers = cnn.layers;
        antsigma = repmat(antsigma, 1, nAnt);
        Dist = 0;
        for a = 1:nAnt
                % 所有filter
                for L = 1:nLayer
                        if strcmp(cnn.layers{L}.type, 'c')
                                for i = 1:numel(cnn.layers{L}.filter)
                                        %% 計算與其他Ant的距離總和→sigma
                                        for b = 1:nAnt
                                                Dist = Dist + abs( Ant{a}.layers{L}.filter{i} - Ant{b}.layers{L}.filter{i} );
                                        end
                                        % 乘上蒸發率除以(nAnt-1)
                                        antsigma(a).layers{L}.sigma{i} = kxi.*Dist/(nAnt-1);
                                end
                        end
                end
        end
        %% New Ant
        NewAnt = [];
        NewAnt{1} = Ant{1};
        NewAnt = repmat(NewAnt, 1, nNewAnt);
        % 第一個Ant不需要做更新
        for a = 1:nNewAnt
                % 所有filter
                for L = 1:nLayer
                        if strcmp(cnn.layers{L}.type, 'c')
                                for i = 1:numel(cnn.layers{L}.filter)
                                        %% 隨機選取一個Ant，取得他所對應的filter及sigma，並根據常態分配在其附近找點
                                        % alpha → Learning rate
                                        p = RouletteWheel(prob);
                                        NewAnt{a}.layers{L}.filter{i} = Ant{p}.layers{L}.filter{i} + alpha .* antsigma(p).layers{L}.sigma{i} * randn;
                                end
                        end
                end
        end
        %% Calculate and Sorting
        % 代入模型計算mse
        Ant = [Ant, NewAnt];
        for a = 1 : nAnt+nNewAnt
                Ant{a} = cnn_fp(Ant{a}, x);
                Ant{a}.error = Ant{a}.output - y;
                Ant{a}.mse = 1/2* sum(Ant{a}.error(:) .^ 2) / size(Ant{a}.error, 2);
                cost(a) = Ant{a}.mse;
        end
        temp = Ant{1}.mse;
        learncurve = [learncurve, temp];
        
        % 依MSE大小做排序
        [~, sorting] = sort(cost);
        Ant = Ant(sorting);
        Ant = Ant(1:nAnt);
end
opts.antalpha = alpha;
end