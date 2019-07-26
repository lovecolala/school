function new_C  = BlockSelection(C, input, All_CenterSigma)
        Num_Rule = size(C, 1);
        Num_Dimension = size(C, 2);
        %% ºâ¥X£]²Ö¥[­È (Rule_Accu)
        Rule_Accu = zeros(1, Num_Rule);
        for i = 1:Num_Rule
                CenterSigma = [];
                for k = 1:size(C, 2)    % Dimension
                        CenterSigma = [CenterSigma; All_CenterSigma{k}.center(C(i, k)), All_CenterSigma{k}.sigma(C(i, k))];
                end
                beta = FirStrgi(input, CenterSigma);
                Rule_Accu(i) = sum(beta);
        end
        average = mean(Rule_Accu);
        %% New Construct Matrix
        t = 1;
        for i = 1:Num_Rule
                if Rule_Accu(i) >= average
                        new_C(t, :) = C(i, :);
                        t = t+1;
                end
        end
end