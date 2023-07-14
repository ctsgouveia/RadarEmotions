function [theclass] = generate_class_3(total_n,T_cond)
%% Generates labels according to a specific test
%%Inputs
% - total_n - number of observations
% - T_cond - specify binary or multiclass test using 'FH' for Fear vs. Happy
% 'FN' for Fear vs. Neutral , 'HN' for Happy vs. Neutral or 'HNF' for multiclass
%%Outputs
% - theclass - vetor com as labels pretendidas
%%
if strcmp(T_cond,'FH')
    str1 = {'Fear '};
    str1_r = repmat(str1,total_n,1);
    str1_r_a = cell2mat(str1_r);
    str2 = {'Happy'};
    str2_r = repmat(str2,total_n,1);
    str2_r_a = cell2mat(str2_r);
    array_class = [str1_r_a; str2_r_a];
    theclass = cellstr(array_class);

elseif strcmp(T_cond,'FN')
    str1 = {'Fear  '};
    str1_r = repmat(str1,total_n,1);
    str1_r_a = cell2mat(str1_r);
    str2 = {'Neutro'};
    str2_r = repmat(str2,total_n,1);
    str2_r_a = cell2mat(str2_r);
    array_class = [str1_r_a; str2_r_a];
    theclass = cellstr(array_class);

elseif strcmp(T_cond,'HN')
    str1 = {'Happy '};
    str1_r = repmat(str1,total_n,1);
    str1_r_a = cell2mat(str1_r);
    str2 = {'Neutro'};
    str2_r = repmat(str2,total_n,1);
    str2_r_a = cell2mat(str2_r);
    array_class = [str1_r_a; str2_r_a];
    theclass = cellstr(array_class);

elseif strcmp(T_cond,'HNF') 
    str1 = {'Happy '};
    str1_r = repmat(str1,total_n,1);
    str1_r_a = cell2mat(str1_r);
    str2 = {'Neutro'};
    str2_r = repmat(str2,total_n,1);
    str2_r_a = cell2mat(str2_r);
    str3 = {'Fear  '};
    str3_r = repmat(str3,total_n,1);
    str3_r_a = cell2mat(str3_r);
    array_class = [str1_r_a; str2_r_a; str3_r_a];
    theclass = cellstr(array_class);
end
end