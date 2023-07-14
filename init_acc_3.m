function [st_FH_cl,st_FN_cl,st_HN_cl,st_HNF_cl] = init_acc_3(repeat_factor)
%% Initialize vectors to store results for each classifier and each test
%%Inputs 
% - repeat_factor - length of the vector with the number of iterations
%%Outputs
% st_XX_cl -  one vector per binary/multiclass condition and performance metric
% where XX is 'FH' fear vs happy, 'FN' fear vs neutro, 'HN' happy vs neutro,
% 'HNF' multiclass
%%
st_FH_cl = zeros(1,repeat_factor);
st_FN_cl = zeros(1,repeat_factor);
st_HN_cl = zeros(1,repeat_factor);
st_HNF_cl = zeros(1,repeat_factor);
end