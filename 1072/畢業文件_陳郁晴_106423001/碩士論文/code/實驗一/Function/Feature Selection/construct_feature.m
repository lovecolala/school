function [feature_table, target_table] = construct_feature(feature_data, target_data, nFeatureDate)
nInput_data = size(feature_data, 1);
nDim_data = size(feature_data, 2);
nFeature = nFeatureDate * nDim_data;

nInput_result = nInput_data - nFeatureDate;


feature_table = zeros(nInput_result, nFeature);
index = (1:nFeatureDate) + (0:nInput_result-1)';

for i = 1:nDim_data
        index_t = (1:nFeatureDate) + (i-1)*nFeatureDate;
        t = feature_data(:, i);
        feature_table(:, index_t) = t(index);
end

target_table = target_data(index(:, end)+1, :);

end