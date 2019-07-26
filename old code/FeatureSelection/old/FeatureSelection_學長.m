function [output,DataMatrix]=FeatureSelection(OriginalData)
%% IIM
% get feature

%caculate Number Of Target
NumberOfTarget=size(OriginalData,2);
%cacurate diff. of all Feature and save as data(t).value
for t=1:NumberOfTarget
        % 台積電
        tsmc=OriginalData(:,t);
        ColOfTarget=31;%(漲跌)+1
        LengthOfData=length(tsmc);
        for jj=1:ColOfTarget
                k=jj;
                for i=1:LengthOfData-ColOfTarget
                        TMP(i,jj)=tsmc(k+1)-tsmc(k);
                        k=k+1;
                end
        end
        data(t).value=TMP;
end

NumberOfTrainPoint=200;
NumberOfTestPoint=size(data,1)-200;

%combine all Feature(no target)
AllFeature=[ ];
for i=1:NumberOfTarget
        TMP=data(i).value(:,1:ColOfTarget-1);
        AllFeature=[AllFeature TMP];
end
%caculate data matrix
DataMatrix=AllFeature;
for i=1:NumberOfTarget
        temp=data(i).value(:,ColOfTarget);
        DataMatrix=[DataMatrix temp];
end

%combine the target(AllFeature+1target)
for i=1:NumberOfTarget
        temp = data(i).value(:,ColOfTarget);
        llMData(i).value=[AllFeature temp];
end

%caculate all IIM for feature selection
%      for i=1:NumberOfTarget
%          IIM(i).value=CaculateIIM(llMData(i).value(1:NumberOfTrainPoint,:));
%      end
% load('IIM4')

%% 計算 gain
%初始化SP
for j=1:NumberOfTarget
        PreSP(j).value(1)=0;
end
%找出當下gain最大的特徵
for j=1:NumberOfTarget
        %初始化索引
        for i = 1:length(IIM(j).value)-1
                index(i)=i;
        end
        count=1;
        %要挑length(IIM(j).value)-1次(每個特徵都要挑)
        for ite=1:length(IIM(j).value)-1
                %算gain值給SPold(j).gain
                %剩下沒被挑過的都要算gain值，所以第ite回合要算length(IIM(j).value)-ite次
                for i=1:length(IIM(j).value)-ite
                        Redundancy=0;
                        i1=index(i);
                        %如果有東西就算冗餘資訊量，沒東西就直接把影響資訊量當作gain
                        if ~PreSP(j).value==0
                                for ii=1:length(PreSP(j).value)
                                        i2=PreSP(j).value(ii);
                                        InformationFi1TOFi2=IIM(j).value(i1,i2);
                                        InformationFi2TOFi1=IIM(j).value(i2,i1);
                                        temp=(InformationFi1TOFi2+InformationFi2TOFi1);
                                        Redundancy=Redundancy+temp;
                                end
                                Information=IIM(j).value(i1,size(IIM(j).value,2));
                                gain=Information-Redundancy;
                        else
                                gain=IIM(j).value(i1,size(IIM(j).value,2));
                        end
                        SPold(j).gain(i)=gain;
                end
                
                %找最大
                tag=1;
                i=1;
                while tag==1
                        if  SPold(j).gain(i)==max(SPold(j).gain)
                                PreSP(j).gain(count)=SPold(j).gain(i);
                                SPold(j).gain(i)=[];
                                PreSP(j).value(count)=index(i);
                                index(i)=[];
                                count=count+1;
                                tag=0;
                        end
                        i=i+1;
                end
        end
end

%把PreSP裡面正的值取出來給SP
for j=1:NumberOfTarget
        count=1;
        for i=1:length(PreSP(j).value)
                if PreSP(j).gain(i)>0
                        SP(j).gain(count)=PreSP(j).gain(i);
                        SP(j).value(count)=PreSP(j).value(i);
                        count=count+1;
                end
        end
end

%% Omega，Omega為每個SP裡的特徵集合(若有重複記一個就好)
%將第一個SP裡的特徵全部加入Omega
Omega=SP(1).value;
LengthOfOmega=length(Omega);
%一開始已經將SP(1)全部加入Omega所以從2開始
for i=2:NumberOfTarget
        for ii=1:length(SP(i).value)
                %any (x==a)如果x中有一個或多個a回傳1，沒則回傳0
                %若Omega中沒有SP(i).value(ii)，將SP(i).value(ii)加入Omega
                if ~any(Omega==SP(i).value(ii))
                        LengthOfOmega=LengthOfOmega+1;
                        Omega(LengthOfOmega)=SP(i).value(ii);
                end
        end
end
%% 計算NOL，NOL為特徵目前出現的累積次數
NOL(1:LengthOfOmega)=0;
for i=1:NumberOfTarget
        for ii=1:LengthOfOmega
                %看Omega(ii)有沒有在SP(i).value裡，有NOL就+1
                if any(SP(i).value==Omega(ii))
                        NOL(ii)=NOL(ii)+1;
                end
        end
end

%% 計算w，w為覆蓋率
for i=1:length(NOL)
        w(i)=NOL(i)/NumberOfTarget;
end
wMean=mean(w);

%% 計算gsum，gsum為特徵在每個SP裡的gain總和
gSum(1:LengthOfOmega)=0;
for i=1:length(Omega)
        for ii=1:NumberOfTarget
                for iii=1:length(SP(ii).value)
                        if Omega(i)==SP(ii).value(iii)
                                gSum(i)=gSum(i)+SP(ii).gain(iii);
                        end
                end
        end
end
gSumMean=mean(gSum);

%% 計算p，p為總貢獻量(W*gsum)
NumberOfFP=0;
for i=1:LengthOfOmega
        p(i)=w(i)*gSum(i);
        if p(i)>wMean*gSumMean
                NumberOfFP=NumberOfFP+1;
                FP.index(NumberOfFP)=Omega(i);
        end
end

%% 定上下界，避免FP選太多或太少
Upper=4;
lower=2;
%將每個FP裡的特徵貢獻值找出來給FP.p
for i=1:length(FP.index)
        for ii=1:LengthOfOmega
                if FP.index(i)==Omega(ii)
                        FP.p(i)=p(ii);
                end
        end
end
%由大到小排序FP的順序
%每回合先把第i個當成最大值，挑出最大值與第i個做交換，做i次即排序完畢
for i=1:length(FP.index)-1
        FPMAX=FP.p(i);
        MAXINDEX=i;
        %找出最大值的位置
        for ii=i:length(FP.index)
                if FP.p(ii)>FPMAX
                        FPMAX=FP.p(ii);
                        MAXINDEX=ii;
                end
        end
        %這回合最大值與第i個p交換
        ptemp=FP.p(i);
        FP.p(i)=FP.p(MAXINDEX);
        FP.p(MAXINDEX)=ptemp;
        
        indextemp=FP.index(i);
        FP.index(i)=FP.index(MAXINDEX);
        FP.index(MAXINDEX)=indextemp;
        
end

%將特徵給output
if length(FP.index)>Upper
        output=FP.index(1:Upper);
else
        output=FP.index;
end
end