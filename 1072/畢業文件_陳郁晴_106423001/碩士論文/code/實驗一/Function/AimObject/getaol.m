function [aol, nThen] = getaol(target, Ra, nTarget)
%% Transfer the target in UDCP
y_mean = mean(target);
y_sigma = std(real(target)) + std(imag(target)) .* sqrt(-1);
target = cGauF(target, y_mean, y_sigma);

%% Subclust
t = complex2real(target, 2, nTarget);
nOutput = size(target, 2);

c = subclust(t, Ra);
nThen = size(c, 1);

%% fuzzy c-means
aol.center = [];
aol.sigma = [];
aol = repmat(aol, 1, nOutput);

for i = 1:nOutput
        temp = target(:, i);
        center = fcm(temp, nThen, [NaN, NaN, NaN, false]);
        temp = temp.';
        sigma = sqrt(sum((temp-center)' .' .* (temp-center), 2) ./ (length(temp)-1));
        
        
%         Re_sigma = sqrt( sum(real(temp-center).^2, 2) ./ length(temp) );
%         Im_sigma = sqrt( sum(imag(temp-center).^2, 2) ./ length(temp) );
        aol(i).center = center;
        aol(i).sigma = sigma;
end

end