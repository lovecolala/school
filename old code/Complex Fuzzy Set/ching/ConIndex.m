function [Rule_index, para] = ConIndex(C, Thenopt1, h)
        para = [];
        Rule_index.center= [];
        tt = 0;
        Num_Dimension = size(C, 2);
        for i = 1:Num_Dimension
                [use_FS, start, index] = unique(C(:, i));               % 針對某一個dimension
                % '*2'是因為有c和sigma, '-1'是為了得到c的index
                index = tt + index*2-1;
                Rule_index.center = [Rule_index.center, index];
                tt = max(max(Rule_index.center))+1;     % 最後一筆sigma的位置
                temp_para = reshape([h{i}.center(use_FS), h{i}.sigma(use_FS)]', 1, []);
                para = [para, temp_para];
        end
        if Thenopt1 == 1
                Num_Thenpara = Num_Dimension + 1;
                Num_Rule = size(C, 1);
                Rule_index.TS = tt+reshape(1:Num_Thenpara*Num_Rule, [], Num_Rule)';
                para = [para, ones(1, numel(Rule_index.TS))];
        end
end