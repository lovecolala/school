function result = calculateBoundary(rate, nClass)
%% Kmeans
nTarget = size(rate, 2);
boundary = zeros(2*nClass, nTarget);
for i = 1:nTarget
        clust = kmeans(rate(:, i), nClass);
        for j = 1:nClass
                clust_rate = rate(clust==j, i);
                index = [2*j-1, 2*j];
                boundary(index, i) = [max(clust_rate); min(clust_rate)];
        end
        boundary(:, i) = sort(boundary(:, i),'descend');
end

%% calculate Boundary
result = zeros(nClass+1, nTarget);
result(1, :) = ones(1, nTarget) .* inf;
result(end, :) = ones(1, nTarget) .* (-inf);
for i = 1:nClass-1
        index = [2*i, 2*i+1];
        result(i+1, :) = mean(boundary(index, :));
end

end