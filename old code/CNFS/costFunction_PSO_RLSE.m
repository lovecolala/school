function answer = costFunction_PSO_RLSE(p_data_pairs,pos,conMatrix,p_index,num_if_part,num_then_part)
    
    data = p_data_pairs;
    data_len = size(data,1);
    data_wid = size(data,2);
    CM_length = size(conMatrix,1);
    index = p_index;
    pos_if = reshape(pos,2,num_if_part/2)';
    aaa.pos=[];
    aaa.cost = 0;
    
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
    
    input = [];
    for i = 1:size(b,1)
        temp = [ones(data_len,1) data(:,1:end-1)]*b(i,1);
        input = [input temp];
    end
    
    
    shita = RLSE(num_then_part,input,data(:,end));
    aaa.pos = shita;
    
    pos_then = reshape(shita,CM_length,data_wid)';
    
    output = (ones(data_len,1)*pos_then(1,:) + data(:,1)*pos_then(2,:) + data(:,2)*pos_then(3,:))*b;

    error = data(:,end) - output ;
    aaa.cost = RMSE(error);
       
answer = aaa;