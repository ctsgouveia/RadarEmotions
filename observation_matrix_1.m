%% Build observation matrix
% Opens signals and compute features
% Builds matrix (N x M -> observations [min] x features)
clear; clc; close all
fs = 100;          % sampling frequency
Emo = 'N';          % select the emotion: 'F' fear, 'H' happy, 'N' neutral

folder_name = 'C:\Users\CarolinaGouveia\OneDrive - Universidade de Aveiro\IEETA_PosDoc\Submissao_artigos\1_Emocoes_Springer3\Submissao_v1\Code_Git_clone';
% folder_name = 'LOCATION';
%% Open 1 minute signals
id = 7;
dbg_balance = 1;
[bR,min_N] = load_dataset_singleID_1(id,dbg_balance);

%% Separate by emotion -> dataset must be balanced
bR_F = bR(1:min_N,:);
bR_H = bR(min_N+1:2*min_N,:);
bR_N = bR((2*min_N)+1:3*min_N,:);

% Observation matrix per emotion
if Emo == 'F'
    BR = bR_F;
elseif Emo == 'H'
    BR = bR_H;
elseif Emo == 'N'
    BR = bR_N;
end

%% Open individual ANN model to compute HR with a reduced error (feature 1)
% Observation matrix of ID07 to use in ANN
baseFilename = sprintf('ANN_HR_matrix_ID%d_1.mat', id);
textFilename{id} = fullfile(folder_name,baseFilename);
if exist(textFilename{id}, 'file')
    s = load(textFilename{id});
    ANN_Obsv = s.Obsv_matrix;
    ANN_Obsv = ANN_Obsv';
else
    error('File does not exist:\n%s', textFilename{id});
end

% ANN Model for ID07
name_Mdl = sprintf("ANN_HR_Mdl_ID%d_1.mat", id);
textFilename{id} = fullfile(folder_name,name_Mdl);
if exist(textFilename{id}, 'file')
    s = load(textFilename{id});
    Mdl_ANN_HR = s.Mdl_AI_R;
else
    error('File does not exist:\n%s', textFilename{id});
end

%% Extract features and build matrix
Obsv_matrix = features_extraction_1(min_N,BR,ANN_Obsv,Mdl_ANN_HR,fs);
fileID = ['matrix_ID_', num2str(id),Emo,'_1'];
save(fileID ,'Obsv_matrix');