function result = calculateErrorRate(error_index, nInput, nTraining)
sum_error = sum(error_index);
nTarget = length(sum_error);

result = zeros(nTarget, 6);
for i = 1:nTarget
        e = sum(error_index(:, i)) ./ nInput;
        eTrain = sum(error_index(1:nTraining, i)) ./ nTraining;
        eTest = sum(error_index(nTraining+1:end, i)) ./ (nInput - nTraining);
        result(i, :) = [e, 1-e, eTrain, 1-eTrain, eTest, 1-eTest];
end

end