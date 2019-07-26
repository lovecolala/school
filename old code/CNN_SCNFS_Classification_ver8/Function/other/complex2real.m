function result = complex2real(matrix, dim)

if dim == 1
        result = zeros(size(matrix, 1)*2, size(matrix, 2));
        result(1:2:end, :) = real(matrix);
        result(2:2:end, :) = imag(matrix);
elseif dim == 2
        result = zeros(size(matrix, 1), size(matrix, 2)*2);
        result(:, 1:2:end) = real(matrix);
        result(:, 2:2:end) = imag(matrix);
end

end