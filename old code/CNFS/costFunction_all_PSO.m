function cost = costFunction_all_PSO(p_data_pairs,pos,conMatrix,p_index,num_if_part)
    
    data = p_data_pairs;
    data_len = size(data,1);
    data_wid = size(data,2);
    CM_length = size(conMatrix,1);
    index = p_index;
    pos_if = reshape(pos(1:num_if_part),2,num_if_part/2)';
    pos_then = reshape(pos(num_if_part+1:end),CM_length,data_wid)';
    
    beta_sec = zeros(CM_length,1);
    for i = 1:CM_length
        % firing strength
        beta = 1;
        for j = 1:data_wid-1
            p = index(j).set(find(index(j).set(:,1)==conMatrix(i,j)),2);
            beta = beta.*cFuzzySet(data(:,j),pos_if(p,:));
        end
        beta_sec(i,1) = sum(beta(:));
    end
    beta_sum = sum(beta_sec(:,1));
    
    b = beta_sec/beta_sum;

    output = (ones(data_len,1)*pos_then(1,:) + data(:,1)*pos_then(2,:) + data(:,2)*pos_then(3,:))*real(b);

    error = data(:,end) - output ;
         
cost = RMSE(error);