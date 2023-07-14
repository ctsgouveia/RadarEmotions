%% Build observation matrix for BIOPAC signals
% Opens signals and compute features
% Builds matrix (N x M -> observations [min] x features)
clear; clc; close all
fs = 100;          % sampling frequency
Emo = 'H';          % select the emotion: 'F' fear, 'H' happy, 'N' neutral

%% Open 1 minute signals
id = 7;
dbg_balance = 1;
[bP,ECG,th,min_N] = load_dataset_singleID_BIOPAC_1(id,dbg_balance);

%% Respiratory signal
%% Separate by emotion -> dataset must be balanced
bP_F = bP(1:min_N,:);
bP_H = bP(min_N+1:2*min_N,:);
bP_N = bP((2*min_N)+1:3*min_N,:);

% Observation matrix per emotion
if Emo == 'F'
    BP = bP_F;
elseif Emo == 'H'
    BP = bP_H;
elseif Emo == 'N'
    BP = bP_N;
end
%% ECG signal
%% Separate by emotion -> dataset must be balanced
ECG_F = ECG(1:min_N,:);
ECG_H = ECG(min_N+1:2*min_N,:);
ECG_N = ECG((2*min_N)+1:3*min_N,:);

% Observation matrix per emotion
if Emo == 'F'
    ECG = ECG_F;
elseif Emo == 'H'
    ECG = ECG_H;
elseif Emo == 'N'
    ECG = ECG_N;
end
%% Extract features and build matrix
Obsv_matrix = features_extraction_BIOPAC_1(min_N,BP,ECG,fs);
fileID = ['matrix_ID_BP_', num2str(id),Emo,'_1'];
save(fileID ,'Obsv_matrix');