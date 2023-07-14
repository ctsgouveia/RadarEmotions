function [bP,ECG,th,min_N] = load_dataset_singleID_BIOPAC_1(ID,dbg_balance)
%% Builds the signals matrix for all conditions
%%Inputs
% - ID - ID of the subject
% - dbg_balance - flag to balance the dataset. All conditions have the same duration.

%%Output
% - bP - respiratory BIOPAC signal
% - ECG - ECG from BIOPAC
% - th - thresholds to detect the R-peak
% - min_N - duration in minutes of the balanced dataset (N). If the dataset is imbalanced, min_N = 0;
%%
folder_name = 'C:\Users\CarolinaGouveia\OneDrive - Universidade de Aveiro\IEETA_PosDoc\Submissao_artigos\1_Emocoes_Springer3\Submissao_v1\Code_Git_clone\Dataset_ID07_condition_1';
%% Fear condition
baseFilename = sprintf('bP_ID0%dF_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    bP_F = s.seg_bP;
else
    error('File does not exist:\n%s', textFilename{ID});
end

baseFilename = sprintf('ECG_ID0%dF_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    ECG_F = s.seg_ECG;
else
    error('File does not exist:\n%s', textFilename{ID});
end

%% Happy condition
baseFilename = sprintf('bP_ID0%dH_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    bP_H = s.seg_bP;
else
    error('File does not exist:\n%s', textFilename{ID});
end

baseFilename = sprintf('ECG_ID0%dH_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    ECG_H = s.seg_ECG;
else
    error('File does not exist:\n%s', textFilename{ID});
end

%% Neutral condition
baseFilename = sprintf('bP_ID0%dN_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    bP_N = s.seg_bP;
else
    error('File does not exist:\n%s', textFilename{ID});
end

baseFilename = sprintf('ECG_ID0%dN_1min.mat', ID);
textFilename{ID} = fullfile(folder_name,baseFilename);
if exist(textFilename{ID}, 'file')
    s = load(textFilename{ID});
    ECG_N = s.seg_ECG;
else
    error('File does not exist:\n%s', textFilename{ID});
end

%% ECG Thresholds
% th = [F H N]
th_aux = [0.3 0.3 0.3;
    0.2 0.6 0.6;
    0.6 0.1 0.6;
    0.1 0.6 0.5;
    0.6 0.2 0.6;
    0.1 0.6 0.6;
    0.4 0.4 0.1;
    0.3 0.3 0.07;
    0.15 0.2 0.15;
    0.4 0.6 0.6;
    0.6 0.6 0.6;
    0.6 0.6 0.6;
    0.6 0.6 0.6;
    0.6 0.6 0.6;
    0.6 0.6 0.6;
    0.2 0.2 0.2;
    0.6 0.6 0.6;
    0.6 0.6 0.6;
    0.6 0.6 0.6;
    0.6 0.6 0.15;
    0.3 0.3 0.3;
    0.6 0.6 0.6];

th_F = th_aux(ID,1)*ones(1,size(ECG_F,1));
th_H = th_aux(ID,2)*ones(1,size(ECG_H,1));
th_N = th_aux(ID,3)*ones(1,size(ECG_N,1));

%% Build dataset
if dbg_balance
    N_F = size(bP_F,1); N_H = size(bP_H,1); N_N = size(bP_N,1);
    min_N = min([N_F N_H N_N]);

    bP = [bP_F(1:min_N,:); bP_H(1:min_N,:); bP_N(1:min_N,:)];
    ECG = [ECG_F(1:min_N,:); ECG_H(1:min_N,:); ECG_N(1:min_N,:)];
    th = [th_F(1:min_N) th_H(1:min_N) th_N(1:min_N)];
else
    min_N = 0;
    bP = [bP_F; bP_H; bP_N];
    ECG = [ECG_F; ECG_H; ECG_N];
    th = [th_F th_H th_N];
end
end

