function cost = outputFunction_PSO_RLSE(p_data_pairs,pos,conMatrix,p_index,num_if_part)
    
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
        pos1 = index(1).set(find(index(1).set(:,1)==conMatrix(i,1)),2);
        pos2 = index(2).set(find(index(2).set(:,1)==conMatrix(i,2)),2);
        beta = cFuzzySet(data',pos_if(pos1,:)).*cFuzzySet(data',pos_if(pos2,:));
        
        beta_sec(i,1) = sum(beta(:));
    end
    beta_sum = sum(beta_sec(:,1));
    
    b = beta_sec/beta_sum;
    
    output = (ones(data_len,1)*pos_then(1,:) + data(:,1)*pos_then(2,:) + data(:,2)*pos_then(3,:))*real(b);
    
    
         
cost = output;