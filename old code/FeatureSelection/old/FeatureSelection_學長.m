function [output,DataMatrix]=FeatureSelection(OriginalData)
%% IIM
% get feature

%caculate Number Of Target
NumberOfTarget=size(OriginalData,2);
%cacurate diff. of all Feature and save as data(t).value
for t=1:NumberOfTarget
        % �x�n�q
        tsmc=OriginalData(:,t);
        ColOfTarget=31;%(���^)+1
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

%% �p�� gain
%��l��SP
for j=1:NumberOfTarget
        PreSP(j).value(1)=0;
end
%��X��Ugain�̤j���S�x
for j=1:NumberOfTarget
        %��l�Ư���
        for i = 1:length(IIM(j).value)-1
                index(i)=i;
        end
        count=1;
        %�n�Dlength(IIM(j).value)-1��(�C�ӯS�x���n�D)
        for ite=1:length(IIM(j).value)-1
                %��gain�ȵ�SPold(j).gain
                %�ѤU�S�Q�D�L�����n��gain�ȡA�ҥH��ite�^�X�n��length(IIM(j).value)-ite��
                for i=1:length(IIM(j).value)-ite
                        Redundancy=0;
                        i1=index(i);
                        %�p�G���F��N�⤾�l��T�q�A�S�F��N������v�T��T�q��@gain
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
                
                %��̤j
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

%��PreSP�̭������Ȩ��X�ӵ�SP
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

%% Omega�AOmega���C��SP�̪��S�x���X(�Y�����ưO�@�ӴN�n)
%�N�Ĥ@��SP�̪��S�x�����[�JOmega
Omega=SP(1).value;
LengthOfOmega=length(Omega);
%�@�}�l�w�g�NSP(1)�����[�JOmega�ҥH�q2�}�l
for i=2:NumberOfTarget
        for ii=1:length(SP(i).value)
                %any (x==a)�p�Gx�����@�өΦh��a�^��1�A�S�h�^��0
                %�YOmega���S��SP(i).value(ii)�A�NSP(i).value(ii)�[�JOmega
                if ~any(Omega==SP(i).value(ii))
                        LengthOfOmega=LengthOfOmega+1;
                        Omega(LengthOfOmega)=SP(i).value(ii);
                end
        end
end
%% �p��NOL�ANOL���S�x�ثe�X�{���ֿn����
NOL(1:LengthOfOmega)=0;
for i=1:NumberOfTarget
        for ii=1:LengthOfOmega
                %��Omega(ii)���S���bSP(i).value�̡A��NOL�N+1
                if any(SP(i).value==Omega(ii))
                        NOL(ii)=NOL(ii)+1;
                end
        end
end

%% �p��w�Aw���л\�v
for i=1:length(NOL)
        w(i)=NOL(i)/NumberOfTarget;
end
wMean=mean(w);

%% �p��gsum�Agsum���S�x�b�C��SP�̪�gain�`�M
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

%% �p��p�Ap���`�^�m�q(W*gsum)
NumberOfFP=0;
for i=1:LengthOfOmega
        p(i)=w(i)*gSum(i);
        if p(i)>wMean*gSumMean
                NumberOfFP=NumberOfFP+1;
                FP.index(NumberOfFP)=Omega(i);
        end
end

%% �w�W�U�ɡA�קKFP��Ӧh�ΤӤ�
Upper=4;
lower=2;
%�N�C��FP�̪��S�x�^�m�ȧ�X�ӵ�FP.p
for i=1:length(FP.index)
        for ii=1:LengthOfOmega
                if FP.index(i)==Omega(ii)
                        FP.p(i)=p(ii);
                end
        end
end
%�Ѥj��p�Ƨ�FP������
%�C�^�X�����i�ӷ��̤j�ȡA�D�X�̤j�ȻP��i�Ӱ��洫�A��i���Y�Ƨǧ���
for i=1:length(FP.index)-1
        FPMAX=FP.p(i);
        MAXINDEX=i;
        %��X�̤j�Ȫ���m
        for ii=i:length(FP.index)
                if FP.p(ii)>FPMAX
                        FPMAX=FP.p(ii);
                        MAXINDEX=ii;
                end
        end
        %�o�^�X�̤j�ȻP��i��p�洫
        ptemp=FP.p(i);
        FP.p(i)=FP.p(MAXINDEX);
        FP.p(MAXINDEX)=ptemp;
        
        indextemp=FP.index(i);
        FP.index(i)=FP.index(MAXINDEX);
        FP.index(MAXINDEX)=indextemp;
        
end

%�N�S�x��output
if length(FP.index)>Upper
        output=FP.index(1:Upper);
else
        output=FP.index;
end
end