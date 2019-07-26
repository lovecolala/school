function [Rule_index, para, Then_index] = ConIndex(C, Thenopt1, h)
        para = [];
        Rule_index= [];
        tt = 0;
        nDim = size(C, 2);
        for i = 1:nDim
                [use_FS, ~, index] = unique(C(:, i));               % 針對某一個dimension
                % '*2'是因為有c和sigma, '-1'是為了得到c的index
                index = tt + index*2-1;
                Rule_index = [Rule_index, index];
                tt = max(max(Rule_index))+1;     % 最後一筆sigma的位置
                temp_para = reshape([h{i}.center(use_FS), h{i}.sigma(use_FS)]', 1, []);
                para = [para, temp_para];
        end
        if Thenopt1 == 1
                nPara_then = nDim + 1;
                Num_Rule = size(C, 1);
                Then_index = tt+reshape( 1:nPara_then * Num_Rule, [], Num_Rule )';
                para = [para, ones(1, numel(Then_index))];
        end
end