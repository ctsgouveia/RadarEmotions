function [bR,min_N] = load_dataset_singleID_1(ID,dbg_balance)
%% Builds the signals matrix for all conditions
%%Inputs
% - ID - ID of the subject
% - dbg_balance - flag to balance the dataset. All conditions have the same duration.

%%Output
% - bR - bio-radar signal for all conditions. N x 6001, where N is the
% duration of each condition (N_F; N_H; N_N)
% - min_N - duration in minutes of the balanced dataset (N). If the dataset is imbalanced, min_N = 0;

%%
folder_name = 'C:\Users\CarolinaGouveia\OneDrive - Universidade de Aveiro\IEETA_PosDoc\Submissao_artigos\1_Emocoes_Springer3\Submissao_v1\Code_Git_clone\Dataset_ID07_condition_1';
%folder_name = 'LOCATION';

%% Fear condition
baseFilename = sprintf('bR_ID0%dF_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    bR_F = s.seg_bR;
else
    error('File does not exist:\n%s', textFilename{ID});
end

%% Happy condition
baseFilename = sprintf('bR_ID0%dH_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    bR_H = s.seg_bR;
else
    error('File does not exist:\n%s', textFilename{ID});
end

%% Neutral condition
baseFilename = sprintf('bR_ID0%dN_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    bR_N = s.seg_bR;
else
    error('File does not exist:\n%s', textFilename{ID});
end

%% Build dataset
if dbg_balance
    N_F = size(bR_F,1); N_H = size(bR_H,1); N_N = size(bR_N,1);
    min_N = min([N_F N_H N_N]);

    bR = [bR_F(1:min_N,:); bR_H(1:min_N,:); bR_N(1:min_N,:)];
else
    min_N = 0;
    bR = [bR_F; bR_H; bR_N];
end
end

