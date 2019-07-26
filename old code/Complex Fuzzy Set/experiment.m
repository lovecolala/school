clc; clear; close all;
record = 10;
table = zeros(record, 3);
thebest = inf;
for kkkk = 1:record
%         PSO_NoSelect;
%         PSO_Select;
        PSORLSE_NoSelect;
%         PSORLSE_Select;
        table(kkkk, :) = [Train_Test_RMSE, All_RMSE];
%         if All_RMSE < thebest
%                 thebest = All_RMSE;
%                 allfigure = findobj(0, 'type', 'figure');
%                 ttt = ['result ', int2str(kkkk)];
%                 saveas(allfigure, ttt);
%         end
end