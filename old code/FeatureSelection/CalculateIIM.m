function [feature_table, IIM] = CalculateIIM(Delta, nFeature)
%% Data pair
nData = length(Delta);
index = repmat(0:nFeature, nData - nFeature, 1) + (1:nData-nFeature)';
feature_table = Delta(index);

Entropy =@(pd, phi, dx) sum(pd .* log10( phi./pd )) .* dx;
Entropy_condition =@(pdx, pdyx, phi, dx, dyx) sum(pdx .* sum(pdyx .* log10( phi ./ pdyx ))) .* dx .* dyx;
%% Influence Information
IIM = zeros(nFeature+1);
for i = 1:nFeature+1
        %% f1+和f1-
        % Feature為正及為負的真實資料
        f1.minus.data = feature_table( feature_table(:, i) < 0, i );
        f1.plus.data = feature_table( feature_table(:, i) >= 0, i );
        
        [ f1.minus.pd_value , f1.minus.pd , f1.minus.dx ] = CalculatePDF( f1.minus.data );
        [ f1.plus.pd_value , f1.plus.pd , f1.plus.dx ] = CalculatePDF( f1.plus.data );
        
        % 計算發生機率
        f1.data = feature_table(:, i);
        f1.pd = fitdist(f1.data, 'kernel');
        f1.minus.prob = cdf( f1.pd, 0 );
        f1.plus.prob = 1 - f1.minus.prob;
        
        for j = 1:nFeature+1
                %% f2
                % event: fitdist下的500個點
                % pd_value: 500個點對應的Pd值
                % pd: fitdist結果（kernal）
                % dx: 500個點之間的距離
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
                
                %% IIM 矩陣
                if i == j
                        IIM(i, j) = 0;
                else
                        IIM(i, j) = I_xy.minus * f1.minus.prob + I_xy.plus * f1.plus.prob;
                end
        end
end
end