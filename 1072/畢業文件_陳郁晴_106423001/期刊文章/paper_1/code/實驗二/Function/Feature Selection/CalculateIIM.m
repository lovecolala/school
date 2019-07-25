function IIM = CalculateIIM(feature_table)
%% Function
Entropy =@(pd, phi, dx) sum(pd .* log10( phi./pd )) .* dx;
Entropy_condition =@(pdx, pdyx, phi, dx, dyx) sum(pdx .* sum(pdyx .* log10( phi ./ pdyx ))) .* dx .* dyx;

%% Influence Information
nFeature = size(feature_table, 2);
IIM = zeros(nFeature);

for i = 1:nFeature
        %% f1+�Mf1-
        % Feature�����ά��t���u����
        f1.minus.data = feature_table( feature_table(:, i) < 0, i );
        f1.plus.data = feature_table( feature_table(:, i) >= 0, i );
        
        [ f1.minus.pd_value , f1.minus.pd , f1.minus.dx ] = CalculatePDF( f1.minus.data );
        [ f1.plus.pd_value , f1.plus.pd , f1.plus.dx ] = CalculatePDF( f1.plus.data );
        
        % �p��o�;��v
        f1.data = feature_table(:, i);
        f1.pd = fitdist(f1.data, 'kernel');
        f1.minus.prob = cdf( f1.pd, 0 );
        f1.plus.prob = 1 - f1.minus.prob;
        
        for j = 1:nFeature
                if i == j
                        continue;
                end
                %% f2
                % event: fitdist�U��500���I
                % pd_value: 500���I������Pd��
                % pd: fitdist���G�]kernal�^
                % dx: 500���I�������Z��
                f2.data = feature_table(:, j);
                [ f2.pd_value , f2.pd , f2.dx ] = CalculatePDF( f2.data );
                
                %% Y|X
                y_x.minus.data = feature_table( feature_table(:, i) < 0 , j );
                y_x.plus.data = feature_table( feature_table(:, i) >= 0 , j );
                [ y_x.minus.pd_value , y_x.minus.pd , y_x.minus.dx ] = CalculatePDF( y_x.minus.data );
                [ y_x.plus.pd_value , y_x.plus.pd , y_x.plus.dx ] = CalculatePDF( y_x.plus.data );
                
                phi = max( [ y_x.minus.pd_value , y_x.plus.pd_value , f2.pd_value , 1 ] );
                
                %% Entropy
                f2.entropy = Entropy( f2.pd_value , phi, f2.dx );
                y_x.minus.entropy = Entropy_condition( f1.minus.pd_value , y_x.minus.pd_value , phi , f1.minus.dx , y_x.minus.dx );
                y_x.plus.entropy = Entropy_condition( f1.plus.pd_value , y_x.plus.pd_value , phi , f1.plus.dx , y_x.plus.dx );
                
                %% I ( X, Y )
                I_xy.minus = f2.entropy - y_x.minus.entropy;
                I_xy.plus = f2.entropy - y_x.plus.entropy;
                
                %% IIM �x�}
                IIM(i, j) = I_xy.minus * f1.minus.prob + I_xy.plus * f1.plus.prob;
                
        end
end
end