function [feature_table, FP] = FeatureSelection(Data, nFeature, nTarget)
%% Calculate IIM
for i = 1:nTarget
        delta{i} = Data(2:end, i) - Data(1:end-1, i);
end
for i = 1:nTarget
        [feature_table{i}, IIM{i}] = CalculateIIM(delta{i}, nFeature);
end
%% gain
% 初始化
for i = 1:nTarget
        SP(i).index = [];
        SP(i).gain = [];
end
g = zeros(nFeature, nTarget);
% 計算gain，並將gain值為正者丟進SP中
for i = 1:nTarget
        for j = 1:nFeature
                % Redundancy
                R = 0;
                if ~isempty(SP(i).index)
                        for k = 1:length(SP(i).index)
                                R = R + IIM{i}(j, SP(i).index(k)) + IIM{i}(SP(i).index(k), j);
                        end
                        R = R / k / 2;
                end
                g(j, i) = IIM{i}(j, end) - R;
                if g(j, i) > 0
                        SP(i).index = [SP(i).index, j];
                        SP(i).gain = [SP(i).gain, g(j, i)];
                end
        end
end
%% Omega (Ω)
% 取出多目標不重複的index
all_SP = [];
for i = 1:nTarget
        all_SP = [all_SP, SP(i).index];
end
Omega = unique(all_SP);
%% covering rate (ω)
nOL = hist(all_SP, Omega);
covering_rate = nOL ./ nTarget;
%% gsum
gsum = zeros(1, length(Omega));
for i = 1:nTarget
        [~, index_1, index_2] = intersect(Omega, SP(i).index);
        gsum(index_1) = gsum(index_1) + SP(i).gain(index_2);
end
%% contribution index (rho) 貢獻
p = covering_rate .* gsum;
% p_threshold = mean(covering_rate) .* mean(gsum);
% [temp, ~] = find(p > p_threshold);
% nTmp = sum(temp);
nTmp = length(Omega);
%% Final Pool
lower = 2; 
if nTarget == 1
        upper = 10;
else
        upper = 20;
end
nFinal = min(max(lower, nTmp), upper);
[~, index] = sort(p,'descend');
index = index(1:nFinal);
FP = Omega(index);
end