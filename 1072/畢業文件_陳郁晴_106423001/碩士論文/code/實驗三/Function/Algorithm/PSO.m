function [Leader_score, Leader_pos, Learning_curve] = PSO(iteration, position, fobj, algoopts, opts)
% fobj = model
% algoopts.nParticle
% algoopts.learning_rate
% algoopts.rlse
% opts for models

%% Transfer option varible
nParticle = algoopts.nParticle;
learning_rate = algoopts.learning_rate;
nDim = length(position);

Learning_curve = zeros(1, iteration);
c0 = 0.85;
c1 = 2;
c2 = 2;
xi1 = rand(1, nDim);
xi2 = rand(1, nDim);

%% Initialize
particle.pos = [];
particle.vel = [];
particle.pbest.pos = [];
particle.pbest.cost = [];

particle = repmat(particle, 1, nParticle);
for i = 1:nParticle
        if i == 1
                particle(i).pos = position;
        else
                particle(i).pos = randn(1, nDim);
        end
        particle(i).vel = zeros(1, nDim);
        particle(i).pbest.pos = particle(i).pos;
        particle(i).pbest.cost = inf;
end
gbest.pos = particle(1).pos;
gbest.cost = inf;

%% Training
for i = 1:iteration
        for j = 1:nParticle
                cost = fobj(particle(j).pos, opts, algoopts.rlse);
                
                if cost < particle(j).pbest.cost
                        particle(j).pbest.pos = particle(j).pos;
                        particle(j).pbest.cost = cost;
                        
                        if cost < gbest.cost
                                gbest.pos = particle(j).pos;
                                gbest.cost = cost;
                        end
                end
                
                particle(j).vel = c0*particle(j).vel + c1*xi1.*(particle(j).pbest.pos - particle(j).pos) + c2*xi2.*(gbest.pos - particle(j).pos);
                particle(j).pos = particle(j).pos + learning_rate .* particle(j).vel;
                
        end
        Learning_curve(i) = gbest.cost;
        
end
Leader_score = gbest.cost;
Leader_pos = gbest.pos;

end