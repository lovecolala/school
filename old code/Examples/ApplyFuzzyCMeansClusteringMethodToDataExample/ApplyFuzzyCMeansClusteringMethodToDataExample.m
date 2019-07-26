%% Cluster Data Using Fuzzy C-Means Clustering
%%
% Load data.

% Copyright 2015 The MathWorks, Inc.

load fcmdata.dat

%%
% Find |2| clusters using fuzzy c-means clustering.
[centers,U] = fcm(fcmdata,2);

%%
% Classify each data point into the cluster with the largest membership
% value.
maxU = max(U);
index1 = find(U(1,:) == maxU);
index2 = find(U(2,:) == maxU);

%%
% Plot the clustered data and cluster centers.
plot(fcmdata(index1,1),fcmdata(index1,2),'ob')
hold on
plot(fcmdata(index2,1),fcmdata(index2,2),'or')
plot(centers(1,1),centers(1,2),'xb','MarkerSize',15,'LineWidth',3)
plot(centers(2,1),centers(2,2),'xr','MarkerSize',15,'LineWidth',3)
hold off
