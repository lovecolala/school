% This function is build for a new version to get new position based on
% PSO.
% Parameters:
%     center => a 1-by-n position which is a vector of n dimensions.
%     A => This parameter will decreased with the increasing iteration
%     number.
%     iter => The iteration number for now.
%     dim => a scalar number. it means dimension number.
%     std => means std. of all position. it will be a 1-by-n vector.

function [new_pos] = random_pos_ver3(center, t, dim, std)
    
    pos = center + randn() .* std*exp(-t) + t*exp(-t);
    new_pos = pos;

end