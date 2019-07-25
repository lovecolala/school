function out = outputFunction(h,a,b)
    h_len = size(h,1);
    h_wid = size(h,2);
    b_len = length(b);
    
    pos = reshape(a,b_len,h_wid)';
    output = (ones(h_len,1)*pos(1,:) + h(:,1)*pos(2,:) + h(:,2)*pos(3,:))*real(b);
     
out = output;