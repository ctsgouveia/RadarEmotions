function [dataset,theclass] = load_Omatrix_2(ids,T_cond)
%% Loads the observation matrix and generate labelled dataset
%%Inputs
% - ids - subject ID - accepts only one subject or a vector of subjects
% - T_cond - Teste condition one aim the generate - 'F','H','N' to use in feature selection; Binary: 'FH', 'FN','HN' and multiclass: 'HNF'
%%Outputs
% - dataset - build dataset
% - theclass - dataset labels
%%
folder_name = 'C:\Users\CarolinaGouveia\OneDrive - Universidade de Aveiro\IEETA_PosDoc\Submissao_artigos\1_Emocoes_Springer3\Submissao_v1\Code_Git_clone\Matrix_wo_baseline_1';
% folder_name = 'LOCATION';
%% F
if length(ids) == 1
    ID = ids;
    baseFilename = sprintf('matrix_NB_%dF.mat', ID);
    textFilename{ID} = fullfile(folder_name,baseFilename);
    if exist(textFilename{ID}, 'file')
        s = load(textFilename{ID});
        Obsv = s.ObsvF_cond;
        ObsvF = Obsv;
    else
        error('File does not exist:\n%s', textFilename{ID});
    end
else
    ID = ids(1);
    baseFilename = sprintf('matrix_NB_%dF.mat', ID);
    textFilename{ID} = fullfile(folder_name,baseFilename);
    if exist(textFilename{ID}, 'file')
        s = load(textFilename{ID});
        Obsv = s.ObsvF_cond;
        ObsvF = Obsv;
    else
        error('File does not exist:\n%s', textFilename{ID});
    end

    for kk = 2:length(ids)
        ID = ids(kk);
        baseFilename = sprintf('matrix_NB_%dF.mat',ID);
        textFilename{ID} = fullfile(folder_name,baseFilename);
        if exist(textFilename{ID}, 'file')
            s = load(textFilename{ID});
            Obsv_aux = s.ObsvF_cond;
            Obsv_aux = Obsv_aux;
        else
            error('File does not exist:\n%s', textFilename{ID});
        end
        ObsvF = [ObsvF; Obsv_aux];
    end
end

%% H
if length(ids) == 1
    ID = ids;
    baseFilename = sprintf('matrix_NB_%dH.mat', ID);
    textFilename{ID} = fullfile(folder_name,baseFilename);
    if exist(textFilename{ID}, 'file')
        s = load(textFilename{ID});
        Obsv = s.ObsvH_cond;
        ObsvH = Obsv;
    else
        error('File does not exist:\n%s', textFilename{ID});
    end
else
    ID = ids(1);
    baseFilename = sprintf('matrix_NB_%dH.mat', ID);
    textFilename{ID} = fullfile(folder_name,baseFilename);
    if exist(textFilename{ID}, 'file')
        s = load(textFilename{ID});
        Obsv = s.ObsvH_cond;
        ObsvH = Obsv;
    else
        error('File does not exist:\n%s', textFilename{ID});
    end

    for kk = 2:length(ids)
        ID = ids(kk);
        baseFilename = sprintf('matrix_NB_%dH.mat',ID);
        textFilename{ID} = fullfile(folder_name,baseFilename);
        if exist(textFilename{ID}, 'file')
            s = load(textFilename{ID});
            Obsv_aux = s.ObsvH_cond;
            Obsv_aux = Obsv_aux;
        else
            error('File does not exist:\n%s', textFilename{ID});
        end
        ObsvH = [ObsvH; Obsv_aux];
    end
end

%% N
if length(ids) == 1
    ID = ids;
    baseFilename = sprintf('matrix_NB_%dN.mat', ID);
    textFilename{ID} = fullfile(folder_name,baseFilename);
    if exist(textFilename{ID}, 'file')
        s = load(textFilename{ID});
        Obsv = s.ObsvN_cond;
        ObsvN = Obsv;
    else
        error('File does not exist:\n%s', textFilename{ID});
    end
else
    ID = ids(1);
    baseFilename = sprintf('matrix_NB_%dN.mat', ID);
    textFilename{ID} = fullfile(folder_name,baseFilename);
    if exist(textFilename{ID}, 'file')
        s = load(textFilename{ID});
        Obsv = s.ObsvN_cond;
        ObsvN = Obsv;
    else
        error('File does not exist:\n%s', textFilename{ID});
    end

    for kk = 2:length(ids)
        ID = ids(kk);
        baseFilename = sprintf('matrix_NB_%dN.mat',ID);
        textFilename{ID} = fullfile(folder_name,baseFilename);
        if exist(textFilename{ID}, 'file')
            s = load(textFilename{ID});
            Obsv_aux = s.ObsvN_cond;
            Obsv_aux = Obsv_aux;
        else
            error('File does not exist:\n%s', textFilename{ID});
        end
        ObsvN = [ObsvN; Obsv_aux];
    end
end

%% Build dataset
if strcmp(T_cond,'FH')
    dataset = [ObsvF; ObsvH];

    % Labels -----------------------------
    total_n = size(ObsvF,1);
    str1 = {'Fear '};
    str1_r = repmat(str1,total_n,1);
    str1_r_a = cell2mat(str1_r);
    str2 = {'Happy'};
    str2_r = repmat(str2,total_n,1);
    str2_r_a = cell2mat(str2_r);
    array_class = [str1_r_a; str2_r_a];
    theclass = cellstr(array_class);

elseif strcmp(T_cond,'FN')
    dataset = [ObsvF; ObsvN];

    % Labels -----------------------------
    total_n = size(ObsvF,1);
    str1 = {'Fear  '};
    str1_r = repmat(str1,total_n,1);
    str1_r_a = cell2mat(str1_r);
    str2 = {'Neutro'};
    str2_r = repmat(str2,total_n,1);
    str2_r_a = cell2mat(str2_r);
    array_class = [str1_r_a; str2_r_a];
    theclass = cellstr(array_class);

elseif strcmp(T_cond,'HN')
    dataset = [ObsvH; ObsvN];

    % Labels -----------------------------
    total_n = size(ObsvF,1);
    str1 = {'Happy '};
    str1_r = repmat(str1,total_n,1);
    str1_r_a = cell2mat(str1_r);
    str2 = {'Neutro'};
    str2_r = repmat(str2,total_n,1);
    str2_r_a = cell2mat(str2_r);
    array_class = [str1_r_a; str2_r_a];
    theclass = cellstr(array_class);

elseif strcmp(T_cond,'HNF') % For multiclass
    dataset = [ObsvH; ObsvN; ObsvF];

elseif T_cond == 'H'        % To use in feature selection
    dataset = ObsvH;
    
elseif T_cond == 'N'        % To use in feature selection
    dataset = ObsvN;
    
elseif T_cond == 'F'        % To use in feature selection
    dataset = ObsvF;
end

% Create class for multiclass or feature selection
if strcmp(T_cond,'H') | strcmp(T_cond,'N') | strcmp(T_cond,'F') | strcmp(T_cond,'HNF')
    total_n = size(ObsvF,1);
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