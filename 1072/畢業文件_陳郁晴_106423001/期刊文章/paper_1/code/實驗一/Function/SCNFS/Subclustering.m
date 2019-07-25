function A = Subclustering(input, Ra)
        nInput = size(input, 1);
        if nInput == 1
                [A.center, A.sigma] = subclust(input', Ra);
                A.sigma = ones(size(A.center)).*A.sigma;
        else
                for i = 1:nInput
                        [A{i}.center, A{i}.sigma] = subclust(input(i, :)', Ra);
                        A{i}.sigma = ones(size(A{i}.center)).*A{i}.sigma;
                end
        end
end