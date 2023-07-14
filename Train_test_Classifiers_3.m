%% Classifier implementation
clear; clc; close all
repeat_factor = 20;     % Number of iterations to divide data into train-test sets

% Open selected features
used_features = load("selected_features_2.mat");
used_features = used_features.selected_features;

% Initialize vectors to store results for each classifier and each test
% Store results of cross-validation, accuracy
% SVM
[CV_FH_SVM,CV_FN_SVM,CV_HN_SVM,CV_HNF_SVM] = init_acc_3(repeat_factor);                       % For Cross-Validation
[accT30_FH_SVM,accT30_FN_SVM,accT30_HN_SVM,accT30_HNF_SVM] = init_acc_3(repeat_factor);       % For accuracy T30
[accTid_FH_SVM,accTid_FN_SVM,accTid_HN_SVM,accTid_HNF_SVM] = init_acc_3(repeat_factor);       % For accuracy Tid
[F1_T30_FH_SVM,F1_T30_FN_SVM,F1_T30_HN_SVM,F1_T30_HNF_SVM] = init_acc_3(repeat_factor);       % For F1_score T30
[F1_Tid_FH_SVM,F1_Tid_FN_SVM,F1_Tid_HN_SVM,F1_Tid_HNF_SVM] = init_acc_3(repeat_factor);       % For F1_score Tid
% KNN
[CV_FH_KNN,CV_FN_KNN,CV_HN_KNN,CV_HNF_KNN] = init_acc_3(repeat_factor);                       % For Cross-Validation
[accT30_FH_KNN,accT30_FN_KNN,accT30_HN_KNN,accT30_HNF_KNN] = init_acc_3(repeat_factor);       % For accuracy T30
[accTid_FH_KNN,accTid_FN_KNN,accTid_HN_KNN,accTid_HNF_KNN] = init_acc_3(repeat_factor);       % For accuracy Tid
[F1_T30_FH_KNN,F1_T30_FN_KNN,F1_T30_HN_KNN,F1_T30_HNF_KNN] = init_acc_3(repeat_factor);       % For F1_score T30
[F1_Tid_FH_KNN,F1_Tid_FN_KNN,F1_Tid_HN_KNN,F1_Tid_HNF_KNN] = init_acc_3(repeat_factor);       % For F1_score Tid
% Random Forest (RF)
[CV_FH_RF,CV_FN_RF,CV_HN_RF,CV_HNF_RF] = init_acc_3(repeat_factor);                           % For Cross-Validation
[accT30_FH_RF,accT30_FN_RF,accT30_HN_RF,accT30_HNF_RF] = init_acc_3(repeat_factor);           % For accuracy T30
[accTid_FH_RF,accTid_FN_RF,accTid_HN_RF,accTid_HNF_RF] = init_acc_3(repeat_factor);           % For accuracy Tid
[F1_T30_FH_RF,F1_T30_FN_RF,F1_T30_HN_RF,F1_T30_HNF_RF] = init_acc_3(repeat_factor);           % For F1_score T30
[F1_Tid_FH_RF,F1_Tid_FN_RF,F1_Tid_HN_RF,F1_Tid_HNF_RF] = init_acc_3(repeat_factor);           % For F1_score Tid

for ii = 1:repeat_factor
    % Select random Test ID and remove from matrix
%     ids = [1 3:20];
%     test_ID = ids(randi(length(ids)));
%     ids(ids == test_ID) = [];
    ids = 7; % Classifier will only work for id 7 as an example
    %% Fear vs. Happy
    % Divide data into train and test sets
    T_cond = 'F';
    [ObsvF,X] = load_Omatrix_2(ids,T_cond);
    ObsvF = ObsvF(:,used_features);
    
    C_F = cvpartition(length(ObsvF),'Holdout',0.3);
    training_F = ObsvF(training(C_F),:);
    test_F = ObsvF(test(C_F),:);

    T_cond = 'H';
    [ObsvH,X] = load_Omatrix_2(ids,T_cond);
    ObsvH = ObsvH(:,used_features);

    C_H = cvpartition(length(ObsvH),'Holdout',0.3);
    training_H = ObsvH(training(C_H),:);
    test_H = ObsvH(test(C_H),:);

    dt_train = [training_F; training_H];
    dt_test = [test_F; test_H];

    % Generate Labels according to the test
    train_class_FH = generate_class_3(size(training_F,1),'FH');
    test_class_FH = generate_class_3(size(test_F,1),'FH');

    % SVM ----------------------------------------------------------------
    cl_FH = fitcsvm(dt_train,train_class_FH,'KernelFunction','rbf', 'ClassNames',{'Fear '; 'Happy'}, 'Standardize',1);
    
    % Cross validation
    cl_CV = crossval(cl_FH,'Leaveout','on');
    CV_FH = kfoldLoss(cl_CV);
    CV_FH_SVM(ii) = (1-CV_FH)*100;
    
    % Test in 30% - accuracy and F1 score
    accT30_FH_SVM(ii) = (1-loss(cl_FH,dt_test,test_class_FH))*100;
    confmat_FH = confusionmat(test_class_FH,predict(cl_FH,dt_test));
    TP = confmat_FH(1,1);
    FP = confmat_FH(1,2);
    FN = confmat_FH(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_FH_SVM(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % Test in new data - accuracy and F1 score
%     [Obsv21FH,X] = load_Omatrix_2(test_ID,'FH');
%     Obsv21FH = Obsv21FH(:,used_features);
%     classTid_FH = generate_class_3(size(Obsv21FH,1)/2,'FH');
%     
%     accTid_FH_SVM(ii) = (1-loss(cl_FH,Obsv21FH,classTid_FH))*100;
%     confmat_FHid = confusionmat(classTid_FH,predict(cl_FH,Obsv21FH));
%     TP = confmat_FHid(1,1);
%     FP = confmat_FHid(1,2);
%     FN = confmat_FHid(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_FH_SVM(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % KNN ----------------------------------------------------------------
    k = 1;
    Mdl_knn_FH = fitcknn(dt_train,train_class_FH,'NumNeighbors',k,'Standardize',1);
    % Cross validation
    cl_knn = crossval(Mdl_knn_FH,'Leaveout','on');
    L_knn = kfoldLoss(cl_knn);
    CV_FH_KNN(ii) = (1-L_knn)*100;
    
    % Test in 30% - accuracy and F1 score
    accT30_FH_KNN(ii) = (1-loss(Mdl_knn_FH,dt_test,test_class_FH))*100;
    confmat_FH = confusionmat(test_class_FH,predict(Mdl_knn_FH,dt_test));
    TP = confmat_FH(1,1);
    FP = confmat_FH(1,2);
    FN = confmat_FH(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_FH_KNN(ii) = (2*((precision*recall)/(precision+recall)))*100;
    
    % Test in new data - accuracy and F1 score
%     accTid_FH_KNN(ii) = (1-loss(Mdl_knn_FH,Obsv21FH,classTid_FH))*100;
%     confmat_FHid = confusionmat(classTid_FH,predict(Mdl_knn_FH,Obsv21FH));
%     TP = confmat_FHid(1,1);
%     FP = confmat_FHid(1,2);
%     FN = confmat_FHid(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_FH_KNN(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % RF -----------------------------------------------------------------
    numTree = 70;
    MdlT_FH = TreeBagger(numTree,dt_train,train_class_FH,'OOBPrediction','On',...
        'Method','classification', 'OOBPredictorImportance', 'On', 'MinLeafSize',5,'PredictorSelect','curvature');

    % Cross validation
    oobErrorBaggedEnsemble = oobError(MdlT_FH);
    misclass_RF = oobErrorBaggedEnsemble(end);
    CV_FH_RF(ii) = (1-misclass_RF)*100;

    % Test in 30% - accuracy and F1 score
    confmat_FH = confusionmat(test_class_FH,predict(MdlT_FH,dt_test));
    accT30_FH_RF(ii) = ((confmat_FH(1,1)+confmat_FH(2,2))/(confmat_FH(1,1)+confmat_FH(2,2)+confmat_FH(1,2)+confmat_FH(2,1)))*100;

    TP = confmat_FH(1,1);
    FP = confmat_FH(1,2);
    FN = confmat_FH(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_FH_RF(ii) = (2*((precision*recall)/(precision+recall)))*100;
    
    % Test in new data - accuracy and F1 score
%     confmat_FHid =confusionmat(classTid_FH,predict(MdlT_FH,Obsv21FH));
%     accTid_FH_RF(ii) = ((confmat_FHid(1,1)+confmat_FHid(2,2))/(confmat_FHid(1,1)+confmat_FHid(2,2)+confmat_FHid(1,2)+confmat_FHid(2,1)))*100;
% 
%     TP = confmat_FHid(1,1);
%     FP = confmat_FHid(1,2);
%     FN = confmat_FHid(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_FH_RF(ii) = (2*((precision*recall)/(precision+recall)))*100;   

    %% Fear vs. Neutral
    % Divide data into train and test sets
    T_cond = 'F';
    [ObsvF,X] = load_Omatrix_2(ids,T_cond);
    ObsvF = ObsvF(:,used_features);

    C_F = cvpartition(length(ObsvF),'Holdout',0.3);
    training_F = ObsvF(training(C_F),:);
    test_F = ObsvF(test(C_F),:);

    T_cond = 'N';
    [ObsvN,X] = load_Omatrix_2(ids,T_cond);
    ObsvN = ObsvN(:,used_features);

    C_N = cvpartition(length(ObsvN),'Holdout',0.3);
    training_N = ObsvN(training(C_N),:);
    test_N = ObsvN(test(C_N),:);

    dt_train = [training_F; training_N];
    dt_test = [test_F; test_N];

    % Generate Labels according to the test
    train_class_FN = generate_class_3(size(training_F,1),'FN');
    test_class_FN = generate_class_3(size(test_F,1),'FN');

    % SVM ----------------------------------------------------------------
    cl_FN = fitcsvm(dt_train,train_class_FN,'KernelFunction','rbf', 'ClassNames',{'Fear  '; 'Neutro'}, 'Standardize',1);
    % Cross validation
    cl_CV = crossval(cl_FN,'Leaveout','on');
    CV_FN = kfoldLoss(cl_CV);
    CV_FN_SVM(ii) = (1-CV_FN)*100;
    
    % Test in 30% - accuracy and F1 score
    accT30_FN_SVM(ii) = (1-loss(cl_FN,dt_test,test_class_FN))*100;
    confmat_FN = confusionmat(test_class_FN,predict(cl_FN,dt_test));
    TP = confmat_FN(1,1);
    FP = confmat_FN(1,2);
    FN = confmat_FN(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_FN_SVM(ii) = (2*((precision*recall)/(precision+recall)))*100;
    
    % Test in new data - accuracy and F1 score
%     [Obsv21FN,X] = load_Omatrix_2(test_ID,'FN');
%     Obsv21FN = Obsv21FN(:,used_features);
%     class21_FN = generate_class_3(size(Obsv21FN,1)/2,'FN');
% 
%     accTid_FN_SVM(ii) = (1-loss(cl_FN,Obsv21FN,class21_FN))*100;
%     confmat_FN2 = confusionmat(class21_FN,predict(cl_FN,Obsv21FN));
%     TP = confmat_FN2(1,1);
%     FP = confmat_FN2(1,2);
%     FN = confmat_FN2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_FN_SVM(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % KNN ----------------------------------------------------------------
    k = 1;
    Mdl_knn_FN = fitcknn(dt_train,train_class_FN,'NumNeighbors',k,'Standardize',1);
    
    % Cross validation
    cl_knn = crossval(Mdl_knn_FN,'Leaveout','on');
    L_knn = kfoldLoss(cl_knn);
    CV_FN_KNN(ii) = (1-L_knn)*100;
    
    % Test in 30% - accuracy and F1 score
    accT30_FN_KNN(ii) = (1-loss(Mdl_knn_FN,dt_test,test_class_FN))*100;
    confmat_FN = confusionmat(test_class_FN,predict(Mdl_knn_FN,dt_test));
    TP = confmat_FN(1,1);
    FP = confmat_FN(1,2);
    FN = confmat_FN(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_FN_KNN(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % Test in new data - accuracy and F1 score
%     accTid_FN_KNN(ii) = (1-loss(Mdl_knn_FN,Obsv21FN,class21_FN))*100;
%     confmat_FN2 = confusionmat(class21_FN,predict(Mdl_knn_FN,Obsv21FN));
%     TP = confmat_FN2(1,1);
%     FP = confmat_FN2(1,2);
%     FN = confmat_FN2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_FN_KNN(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % RF -----------------------------------------------------------------
    numTree = 70;
    MdlT_FN = TreeBagger(numTree,dt_train,train_class_FN,'OOBPrediction','On',...
        'Method','classification', 'OOBPredictorImportance', 'On', 'MinLeafSize',5,'PredictorSelect','curvature');

    % Cross validation
    oobErrorBaggedEnsemble = oobError(MdlT_FN);
    misclass_RF = oobErrorBaggedEnsemble(end);
    CV_FN_RF(ii) = (1-misclass_RF)*100;
    
    % Test in 30% - accuracy and F1 score
    confmat_FN = confusionmat(test_class_FN,predict(MdlT_FN,dt_test));
    accT30_FN_RF(ii) = ((confmat_FN(1,1)+confmat_FN(2,2))/(confmat_FN(1,1)+confmat_FN(2,2)+confmat_FN(1,2)+confmat_FN(2,1)))*100;
    
    TP = confmat_FN(1,1);
    FP = confmat_FN(1,2);
    FN = confmat_FN(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_FN_RF(ii) = (2*((precision*recall)/(precision+recall)))*100;
   
    % Test in new data - accuracy and F1 score
%     confmat_FN2 = confusionmat(class21_FN,predict(MdlT_FN,Obsv21FN));
%     accTid_FN_RF(ii) = ((confmat_FN2(1,1)+confmat_FN2(2,2))/(confmat_FN2(1,1)+confmat_FN2(2,2)+confmat_FN2(1,2)+confmat_FN2(2,1)))*100;
% 
%     TP = confmat_FN2(1,1);
%     FP = confmat_FN2(1,2);
%     FN = confmat_FN2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_FN_RF(ii) = (2*((precision*recall)/(precision+recall)))*100;

    %% Happy vs. Neutral -------------------------------------------------
    % Divide data into train and test sets
    T_cond = 'H';
    [ObsvH,X] = load_Omatrix_2(ids,T_cond);
    ObsvH = ObsvH(:,used_features);

    C_H = cvpartition(length(ObsvH),'Holdout',0.3);
    training_H = ObsvH(training(C_H),:);
    test_H = ObsvH(test(C_H),:);

    T_cond = 'N';
    [ObsvN,X] = load_Omatrix_2(ids,T_cond);
    ObsvN = ObsvN(:,used_features);

    C_N = cvpartition(length(ObsvN),'Holdout',0.3);
    training_N = ObsvN(training(C_N),:);
    test_N = ObsvN(test(C_N),:);

    dt_train = [training_H; training_N];
    dt_test = [test_H; test_N];

    % Generate Labels according to the test
    train_class_HN = generate_class_3(size(training_H,1),'HN');
    test_class_HN = generate_class_3(size(test_H,1),'HN');

    % SVM -----------------------------------------------------------------
    cl_HN = fitcsvm(dt_train,train_class_HN,'KernelFunction','rbf', 'ClassNames',{'Happy '; 'Neutro'}, 'Standardize',1);
    
    % Cross validation
    cl_CV = crossval(cl_HN,'Leaveout','on');
    CV_HN = kfoldLoss(cl_CV);
    CV_HN_SVM(ii) = (1-CV_HN)*100;
    
    % Test in 30% - accuracy and F1 score
    accT30_HN_SVM(ii) = (1-loss(cl_HN,dt_test,test_class_HN))*100;
    confmat_HN = confusionmat(test_class_HN,predict(cl_HN,dt_test));
    TP = confmat_HN(1,1);
    FP = confmat_HN(1,2);
    FN = confmat_HN(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_HN_SVM(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % Test in new data - accuracy and F1 score
%     [Obsv21HN,X] = load_Omatrix_2(test_ID,'HN');
%     Obsv21HN = Obsv21HN(:,used_features);
%     class21_HN = generate_class_3(size(Obsv21HN,1)/2,'HN');
% 
%     accTid_HN_SVM(ii) = (1-loss(cl_HN,Obsv21HN,class21_HN))*100;
%     confmat_HN2 = confusionmat(class21_HN,predict(cl_HN,Obsv21HN));
%     TP = confmat_HN2(1,1);
%     FP = confmat_HN2(1,2);
%     FN = confmat_HN2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_HN_SVM(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % KNN -----------------------------------------------------------------
    k = 1;
    Mdl_knn_HN = fitcknn(dt_train,train_class_HN,'NumNeighbors',k,'Standardize',1);
    
    % Cross validation
    cl_knn = crossval(Mdl_knn_HN,'Leaveout','on');
    L_knn = kfoldLoss(cl_knn);
    CV_HN_KNN(ii) = (1-L_knn)*100;

    % Test in 30% - accuracy and F1 score
    accT30_HN_KNN(ii) = (1-loss(Mdl_knn_HN,dt_test,test_class_HN))*100;
    confmat_HN = confusionmat(test_class_HN,predict(Mdl_knn_HN,dt_test));
    TP = confmat_HN(1,1);
    FP = confmat_HN(1,2);
    FN = confmat_HN(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_HN_KNN(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % Test in new data - accuracy and F1 score
%     accTid_HN_KNN(ii) = (1-loss(Mdl_knn_HN,Obsv21HN,class21_HN))*100;
%     confmat_HN2 = confusionmat(class21_HN,predict(Mdl_knn_HN,Obsv21HN));
%     TP = confmat_HN2(1,1);
%     FP = confmat_HN2(1,2);
%     FN = confmat_HN2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_HN_KNN(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % RF ------------------------------------------------------------------
    numTree = 70;
    MdlT_HN = TreeBagger(numTree,dt_train,train_class_HN,'OOBPrediction','On',...
        'Method','classification', 'OOBPredictorImportance', 'On', 'MinLeafSize',5,'PredictorSelect','curvature');

    % Cross validation
    oobErrorBaggedEnsemble = oobError(MdlT_HN);
    misclass_RF = oobErrorBaggedEnsemble(end);
    CV_HN_RF(ii) = (1-misclass_RF)*100;

    % Test in 30% - accuracy and F1 score
    confmat_HN = confusionmat(test_class_HN,predict(MdlT_HN,dt_test));
    accT30_HN_RF(ii) = ((confmat_HN(1,1)+confmat_HN(2,2))/(confmat_HN(1,1)+confmat_HN(2,2)+confmat_HN(1,2)+confmat_HN(2,1)))*100;

    TP = confmat_HN(1,1);
    FP = confmat_HN(1,2);
    FN = confmat_HN(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_HN_RF(ii) = (2*((precision*recall)/(precision+recall)))*100;
    
    % Test in new data - accuracy and F1 score
%     confmat_HN2 = confusionmat(class21_HN,predict(MdlT_HN,Obsv21HN));
%     accTid_HN_RF(ii) = ((confmat_HN2(1,1)+confmat_HN2(2,2))/(confmat_HN2(1,1)+confmat_HN2(2,2)+confmat_HN2(1,2)+confmat_HN2(2,1)))*100;
%     
%     TP = confmat_HN2(1,1);
%     FP = confmat_HN2(1,2);
%     FN = confmat_HN2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_HN_RF(ii) = (2*((precision*recall)/(precision+recall)))*100;

    %% Multiclass
    % Divide data into train and test sets
    T_cond = 'H';
    [ObsvH,X] = load_Omatrix_2(ids,T_cond);
    ObsvH = ObsvH(:,used_features);

    C_H = cvpartition(length(ObsvH),'Holdout',0.3);
    training_H = ObsvH(training(C_H),:);
    test_H = ObsvH(test(C_H),:);

    T_cond = 'N';
    [ObsvN,X] = load_Omatrix_2(ids,T_cond);
    ObsvN = ObsvN(:,used_features);

    C_N = cvpartition(length(ObsvN),'Holdout',0.3);
    training_N = ObsvN(training(C_N),:);
    test_N = ObsvN(test(C_N),:);

    T_cond = 'F';
    [ObsvF,X] = load_Omatrix_2(ids,T_cond);
    ObsvF = ObsvF(:,used_features);

    C_F = cvpartition(length(ObsvF),'Holdout',0.3);
    training_F = ObsvF(training(C_F),:);
    test_F = ObsvF(test(C_F),:);

    dt_train = [training_H; training_N; training_F];
    dt_test = [test_H; test_N; test_F];

    % Generate Labels according to the test
    train_class_HNF = generate_class_3(size(training_H,1),'HNF');
    test_class_HNF = generate_class_3(size(test_H,1),'HNF');

    % SVM ----------------------------------------------------------------
    t = templateSVM('KernelFunction','rbf','Standardize',1);
    Mdl = fitcecoc(dt_train,train_class_HNF,'Learners',t,...
        'ClassNames',{'Happy '; 'Neutro'; 'Fear  '});
    
    % Cross validation
    cl_CV = crossval(Mdl,'Leaveout','on');
    CV_HNF = kfoldLoss(cl_CV);
    CV_HNF_SVM(ii) = (1-CV_HNF)*100;

    % Test in 30% - accuracy and F1 score
    accT30_HNF_SVM(ii) = (1-loss(Mdl,dt_test,test_class_HNF))*100;
    confmat_HNF = confusionmat(test_class_HNF,predict(Mdl,dt_test));
    TP = confmat_HNF(1,1);
    FP = confmat_HNF(1,2);
    FN = confmat_HNF(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_HNF_SVM(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % Test in new data - accuracy and F1 score
%     [Obsv21HNF,X] = load_Omatrix_2(test_ID,'HNF');
%     Obsv21HNF = Obsv21HNF(:,used_features);
%     class21_HNF = generate_class_3(size(Obsv21HNF,1)/3,'HNF');
% 
%     accTid_HNF_SVM(ii) = (1-loss(Mdl,Obsv21HNF,class21_HNF))*100;
%     confmat_HNF2 = confusionmat(class21_HNF,predict(Mdl,Obsv21HNF));
%     TP = confmat_HNF2(1,1);
%     FP = confmat_HNF2(1,2);
%     FN = confmat_HNF2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_HNF_SVM(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % KNN -----------------------------------------------------------------
    k = 1;
    Mdl_knn_HNF = fitcknn(dt_train,train_class_HNF,'NumNeighbors',k,'Standardize',1);
    
    % Cross validation
    cl_knn = crossval(Mdl_knn_HNF,'Leaveout','on');
    L_knn = kfoldLoss(cl_knn);
    CV_HNF_KNN(ii) = (1-L_knn)*100;

    % Test in 30% - accuracy and F1 score
    accT30_HNF_KNN(ii) = (1-loss(Mdl_knn_HNF,dt_test,test_class_HNF))*100;
    confmat_HNF = confusionmat(test_class_HNF,predict(Mdl_knn_HNF,dt_test));
    TP = confmat_HNF(1,1);
    FP = confmat_HNF(1,2);
    FN = confmat_HNF(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_HNF_KNN(ii) = (2*((precision*recall)/(precision+recall)))*100;
    
    % Test in new data - accuracy and F1 score
%     accTid_HNF_KNN(ii) = (1-loss(Mdl_knn_HNF,Obsv21HNF,class21_HNF))*100;
%     confmat_HNF2 = confusionmat(class21_HNF,predict(Mdl_knn_HNF,Obsv21HNF));
%     TP = confmat_HNF2(1,1);
%     FP = confmat_HNF2(1,2);
%     FN = confmat_HNF2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_HNF_KNN(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % RF -----------------------------------------------------------------
    numTree = 70;
    MdlT_HNF = TreeBagger(numTree,dt_train,train_class_HNF,'OOBPrediction','On',...
        'Method','classification', 'OOBPredictorImportance', 'On', 'MinLeafSize',5,'PredictorSelect','curvature');

    % Cross validation
    oobErrorBaggedEnsemble = oobError(MdlT_HNF);
    misclass_RF = oobErrorBaggedEnsemble(end);
    CV_HNF_RF(ii) = (1-misclass_RF)*100;

    % Test in 30% - accuracy and F1 score
    confmat_HNF = confusionmat(test_class_HNF,predict(MdlT_HNF,dt_test));
    accT30_HNF_RF(ii) = ((confmat_HNF(1,1)+confmat_HNF(2,2)+confmat_HNF(3,3))/...
        (sum(sum(confmat_HNF))))*100;
    
    TP = confmat_HNF(1,1);
    FP = confmat_HNF(1,2);
    FN = confmat_HNF(2,1);

    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1_T30_HNF_RF(ii) = (2*((precision*recall)/(precision+recall)))*100;

    % Test in new data - accuracy and F1 score
%     confmat_HNF2 = confusionmat(class21_HNF,predict(MdlT_HNF,Obsv21HNF));
%     accTid_HNF_RF(ii) = ((confmat_HNF2(1,1)+confmat_HNF2(2,2)+confmat_HNF2(3,3))/...
%         (sum(sum(confmat_HNF2))))*100;
% 
%     TP = confmat_HNF2(1,1);
%     FP = confmat_HNF2(1,2);
%     FN = confmat_HNF2(2,1);
% 
%     precision = TP/(TP+FP);
%     recall = TP/(TP+FN);
%     F1_Tid_HNF_RF(ii) = (2*((precision*recall)/(precision+recall)))*100;

end