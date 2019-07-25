%% This fuction must combines with function allcomb 
% nums_inputs : A constant. Numbers of inputs.
% cens_inputs : A M*1 matrix. The ith column means numbers of centers of the ith inputs.

function A = constructMatrix(num_inputs,num_cen_inputs)
    a = cell([1,num_inputs]);
    for i = 1:num_inputs
        temp = [1:num_cen_inputs(i)];
        a{1,i} = temp;
    end
A = allcomb(a{:});