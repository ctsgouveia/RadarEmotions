%% Selects features using statistical study Fig. 7
clear; close all; clc
dbg_normal = 1;     % flag to verify if features have normal distribution

%% Create dataset
ID = 7;
T_cond = 'F';
[OF,X] = load_Omatrix_2(ID,T_cond);
ObsvF = OF;
T_cond = 'H';
[OH,X] = load_Omatrix_2(ID,T_cond);
ObsvH = OH;
T_cond = 'N';
[ON,X] = load_Omatrix_2(ID,T_cond);
ObsvN = ON;

ObsvT = [ObsvF; ObsvH; ObsvN];

%% Verify if ANOVA residuals have normal distribution
if dbg_normal
    rng('default');
    groups = {'F' 'H' 'N'};
    result_kstest = zeros(1,60);        % Initialize test result vector
    result_adtest = zeros(1,60);        % Initialize test result vector
    result_lillietest = zeros(1,60);    % Initialize test result vector
    count = zeros(1,60);                % Initialize counter for majority vote
     for ii = 1:60
        % For each feature compute ANOVA residuals 
        XX = [ObsvF(:,ii) ObsvH(:,ii) ObsvN(:,ii)];
        [p,tbl,stats] = anova1(XX,groups,'off');
        % ANOVA residuals
        XX_res = XX-stats.means;    
        XX_res_V = reshape(XX_res,1,[]);
        % Compute Kolmogorov-Smirnov, Anderson-Darling and Lilliefors test
        result_kstest(ii) = kstest(XX_res_V);
        result_adtest(ii) = adtest(XX_res_V);
        result_lillietest(ii) = lillietest(XX_res_V);
        % Verify the majority vote
        if result_kstest(ii) == 1
            count(ii) = count(ii)+1;
        else
            count(ii) = count(ii);
        end
        if result_adtest(ii) == 1
            count(ii) = count(ii)+1;
        else
            count(ii) = count(ii);
        end
        if result_lillietest(ii) == 1
            count(ii) = count(ii)+1;
        else
            count(ii) = count(ii);
        end
     end
end

% By using the complete dataset only one feature did not reject the null 
% hypothesis. Therefore the remain analysis will be conducted using the
% Kruskal-Wallis method instead of ANOVA.

%% Evaluate p-value with Kruskal-Wallis
rng('default');
p = zeros(1,60);
groups = {'F' 'H' 'N'};
for ii = 1:60
    XX = [ObsvF(:,ii) ObsvH(:,ii) ObsvN(:,ii)];
    [p(ii),tbl,stats] = kruskalwallis(XX,groups,'off');
end

I = find(p < 0.05);

%% Pairwise T-test with one-step Bonferroni to evaluate feature differentiation between classes
% Only features with p < 0.05 are now considered

p_FH = zeros(1,length(I));      % Initialize features vector per class pair
p_FN = zeros(1,length(I));      % Initialize features vector per class pair
p_HN = zeros(1,length(I));      % Initialize features vector per class pair

for jj = 1:length(I)
    XX = [ObsvF(:,I(jj)) ObsvH(:,I(jj)) ObsvN(:,I(jj))];
    [p,tbl,stats] = kruskalwallis(XX,groups,'off');
    c = multcompare(stats,'CType','bonferroni','Display','off');
    tbl2 = array2table(c,"VariableNames", ...
        ["Group A","Group B","Lower Limit","A-B","Upper Limit","P-value"]);
    p_FH(jj) = c(1,6);
    p_FN(jj) = c(2,6);
    p_HN(jj) = c(3,6);
end

% Sort in p-value ascend order and select features with p < 0.05 for each
% binary case
[B,Index] = sort(p_FH,'ascend');
f_FH = I(Index);
p_FH_order = B;
p_FH_order = p_FH_order(p_FH_order<0.05);
f_FH = f_FH(p_FH_order<0.05);

[B,Index] = sort(p_FN,'ascend');
f_FN = I(Index);
p_FN_order = B;
p_FN_order = p_FN_order(p_FN_order<0.05);
f_FN = f_FN(p_FN_order<0.05);

[B,Index] = sort(p_HN,'ascend');
f_HN = I(Index);
p_HN_order = B;
p_HN_order = p_HN_order(p_HN_order<0.05);
f_HN = f_HN(p_HN_order<0.05);

%% Create priority queue
% Cut priority list length to the minimum list length 
% (the excess is neglected due to the high p-value and repeated entry)
T_lengths = [length(f_FH) length(f_FN) length(f_HN)];
L = min(T_lengths);
f_queue = zeros(1,3*L);                 % Initialize priority queue
f_queue = [f_FH(1) f_FN(1) f_HN(1)];    % Initialize the first three entries of queue

for k = 2:L
    f_queue = [f_queue f_FH(k) f_FN(k) f_HN(k)];
end

f_queue_priority = unique(f_queue,'stable'); % Remove repeated features keeping queue order          

%% Correlation matrix
f_queue = unique(f_queue);          % Remove repeated features and set features in order by name
selF_observ = ObsvT(:,f_queue);     % Observation matrix correspondent to the selected features
CM = corrplot(selF_observ);         % Compute correlation matrix

% Display Correlation Matrix
figure; imagesc(CM);
h = gca;
h.XTick = 1:length(f_queue);
h.YTick = 1:length(f_queue);
h.XTickLabel = f_queue;
h.YTickLabel= f_queue;
colorbar
colormap('hot')

%% Remove redundant features according to priority list
Nf_queue = 0;       % Initialize select features vector
for k_row = 1:length(CM)
    for k_column = 1:length(CM)
        if k_row == k_column
            continue;
        elseif k_column < k_row
            continue;
        elseif abs(CM(k_row,k_column)) > 0.7
            Index_r = find(f_queue_priority == f_queue(k_row));
            Index_c = find(f_queue_priority == f_queue(k_column));
            if Index_r < Index_c
                Nf_queue = [Nf_queue f_queue(k_column)];
            else
                Nf_queue = [Nf_queue f_queue(k_row)];
                break;
            end
        end
    end
end

selected_features_to_remove = Nf_queue(Nf_queue~=0);
Acommon = intersect(f_queue,selected_features_to_remove);

selected_features = setxor(f_queue,Acommon);