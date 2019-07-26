function output = real2class(output, nClass, nTarget)
for i = 1:nTarget
        index = (1:nClass)+(i-1)*nClass;
        [~, b] = max(output.real(:, index), [], 2);
        output.classification(:, i) = b;
end
end