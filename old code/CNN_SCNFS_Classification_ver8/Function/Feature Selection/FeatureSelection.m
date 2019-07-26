function FP = FeatureSelection(feature_table, IIM, nTarget)
%% gain
% ��l��
SP.index = [];
SP.gain = [];
SP = repmat(SP, 1, nTarget);

nFeature = size(feature_table, 2) - nTarget;
g = zeros(nFeature, nTarget);
% �p��gain�A�ñNgain�Ȭ����̥�iSP��

for i = 1:nTarget
        for j = 1:nFeature
                % Redundancy
                R = 0;
                if ~isempty(SP(i).index)
                        for k = 1:length(SP(i).index)
                                R = R + IIM(j, SP(i).index(k)) + IIM(SP(i).index(k), j);
                        end
                        R = R / k / 2;
                end
                g(j, i) = IIM(j, end-nTarget+i) - R;
                if g(j, i) > 0
                        SP(i).index = [SP(i).index, j];
                        SP(i).gain = [SP(i).gain, g(j, i)];
                end
        end
end

%% Omega (�[)
% ���X�h�ؼФ����ƪ�index
all_SP = [];
for i = 1:nTarget
        all_SP = [all_SP, SP(i).index];
end
Omega = unique(all_SP);

%% covering rate (�s)
nOL = hist(all_SP, Omega);
covering_rate = nOL ./ nTarget;

%% gsum
gsum = sum(IIM(Omega, (1:nTarget)+nFeature), 2)';

%% contribution index (rho) �^�m
p = covering_rate .* gsum;

%% Threshold
pth = 0;
ntmp = sum(p>pth);
nFP = floor(sqrt(ntmp))^2;

%% Final Pool
[~, index] = sort(p, 'descend');
FP = Omega(index);
FP = FP(1:nFP);

end