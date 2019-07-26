function cnn_output = cnn_transfer(cnn)
cnn_output = [];
for i = 1:numel(cnn.layers{end}.output)
        sizeA = size(cnn.layers{end}.output{i});
        t = reshape(cnn.layers{end}.output{i}, [], sizeA(end));
        cnn_output = [cnn_output; t];
end

end