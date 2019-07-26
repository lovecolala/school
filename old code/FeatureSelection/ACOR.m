function [Ant, opts, learncurve] = ACOR(cnn, Ant, prob, x, y, opts)
% �Ĥ@���i�Ӯ�Ant�u���@��
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
                % �Ҧ�filter
                for L = 1:nLayer
                        if strcmp(cnn.layers{L}.type, 'c')
                                for i = 1:numel(cnn.layers{L}.filter)
                                        %% �p��P��LAnt���Z���`�M��sigma
                                        for b = 1:nAnt
                                                Dist = Dist + abs( Ant{a}.layers{L}.filter{i} - Ant{b}.layers{L}.filter{i} );
                                        end
                                        % ���W�]�o�v���H(nAnt-1)
                                        antsigma(a).layers{L}.sigma{i} = kxi.*Dist/(nAnt-1);
                                end
                        end
                end
        end
        %% New Ant
        NewAnt = [];
        NewAnt{1} = Ant{1};
        NewAnt = repmat(NewAnt, 1, nNewAnt);
        % �Ĥ@��Ant���ݭn����s
        for a = 1:nNewAnt
                % �Ҧ�filter
                for L = 1:nLayer
                        if strcmp(cnn.layers{L}.type, 'c')
                                for i = 1:numel(cnn.layers{L}.filter)
                                        %% �H������@��Ant�A���o�L�ҹ�����filter��sigma�A�îھڱ`�A���t�b�������I
                                        % alpha �� Learning rate
                                        p = RouletteWheel(prob);
                                        NewAnt{a}.layers{L}.filter{i} = Ant{p}.layers{L}.filter{i} + alpha .* antsigma(p).layers{L}.sigma{i} * randn;
                                end
                        end
                end
        end
        %% Calculate and Sorting
        % �N�J�ҫ��p��mse
        Ant = [Ant, NewAnt];
        for a = 1 : nAnt+nNewAnt
                Ant{a} = cnn_fp(Ant{a}, x);
                Ant{a}.error = Ant{a}.output - y;
                Ant{a}.mse = 1/2* sum(Ant{a}.error(:) .^ 2) / size(Ant{a}.error, 2);
                cost(a) = Ant{a}.mse;
        end
        temp = Ant{1}.mse;
        learncurve = [learncurve, temp];
        
        % ��MSE�j�p���Ƨ�
        [~, sorting] = sort(cost);
        Ant = Ant(sorting);
        Ant = Ant(1:nAnt);
end
opts.antalpha = alpha;
end