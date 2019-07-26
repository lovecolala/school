function result = real2complex(matrix, dim)
if dim == 1
        result = matrix(1:2:end, :) + matrix(2:2:end, :) .* sqrt(-1);
elseif dim == 2
        result = matrix(:, 1:2:end) + matrix(:, 2:2:end) .* sqrt(-1);
end
end