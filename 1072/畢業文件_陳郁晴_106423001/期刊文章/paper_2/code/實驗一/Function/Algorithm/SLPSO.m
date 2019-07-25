function [Leader_score, Leader_pos, Learning_curve] = SLPSO(iteration, position, fobj, algoopts, opts)
% fobj = model
% algoopts.nBaseParticle
% opts for models

%% Transfer option varible
nBaseParticle = algoopts.nBaseParticle;
nDim = length(position);

Learning_curve = zeros(1, iteration);
nParticle = nBaseParticle + floor(nDim/10);
c3 = nDim / nBaseParticle * 0.01;
probLearn = (1 - (0:(nParticle-1))'./nParticle).^log(sqrt(ceil(nDim/nBaseParticle)));

%% Initialize
particle = zeros(nParticle, nDim);
cost = zeros(1, nParticle);
for i = 1:nParticle
        if i == 1
                particle(i, :) = position;
        else
                particle(i, :) = randn(1, nDim);
        end
        cost(i) = fobj(particle(i, :), opts, algoopts.rlse);
end
[cost, sorting] = sort(cost, 'descend');        % 逼cost程程
particle = particle(sorting, :);
veloxity = zeros(nParticle, nDim);      % ﹍
best.pos = particle(end, :);
best.cost = cost(end);

%% Training
for i = 1:iteration
        % Demonstrator
        demoIndexMask = (1:nParticle)';
        demoIndex = demoIndexMask + ceil(rand(nParticle, nDim) .* (nParticle - demoIndexMask));
        demonstrator = particle;
        % ゴ睹采ぃ蝴程玂痙程
        for j = 1:nDim
                demonstrator(:, j) = particle(demoIndex(:, j), j);
        end
        
        % Collective behavior
        center = mean(particle);
        
        % Random matrix
        randco1 = rand(nParticle, nDim);
        randco2 = rand(nParticle, nDim);
        randco3 = rand(nParticle, nDim);
        
        % Social learning
        lpmask = rand(nParticle, 1) < probLearn;
        lpmask(end) = 0;
        
        v1 = randco1 .* veloxity + randco2 .* (demonstrator - particle) + c3*randco3.*(center - particle);
        p1 = particle + v1;
        
        veloxity = lpmask.*v1 + (~lpmask).*veloxity;
        particle = lpmask.*p1 + (~lpmask).*particle;
        
        for j = 1:nParticle-1
                cost(j) = fobj(particle(j, :), opts, algoopts.rlse);
        end
        [cost, sorting] = sort(cost, 'descend');
        particle = particle(sorting, :);
        veloxity = veloxity(sorting, :);
        best.pos = particle(end, :);
        best.cost = cost(end);
        
        Learning_curve(i) = best.cost;
        
end
Leader_score = best.cost;
Leader_pos = best.pos;

end