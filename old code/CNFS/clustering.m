% ===== clustering =====
function A = clustering(p_data_pairs,p_num_inputs,p_clusterInfluenceRange)
       
     
    % parameter of subclust
    data_pairs = p_data_pairs;
    num_inputs = p_num_inputs;
    clusterInfluenceRange = p_clusterInfluenceRange;
    im = sqrt(-1);
    
    % Parameters of inputs
    class_cluster.center = [];
    class_cluster.std = [];
    class_cluster.numOfCenter = [];
    class_cluster.GMF = [];

    % Constructor
    cluster = repmat(class_cluster, num_inputs, 1);

    % Computing
    for i = 1:num_inputs
        % Initialize the parameters
        [C S] = subclust(data_pairs(:,i),clusterInfluenceRange);
        cluster(i).center = C;
        cluster(i).numOfCenter = length(cluster(i).center);
        cluster(i).std = ones(cluster(i).numOfCenter,1)*S;
        
        % Calculate the MFs
        for j = 1:cluster(i).numOfCenter
            cluster(i).GMF(j,:) = cFuzzySet(data_pairs(:,i),[cluster(i).center(j) cluster(i).std(j)]);
        end
    end 

A = cluster;