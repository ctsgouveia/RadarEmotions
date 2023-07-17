# RadarEmotions
Herein are the MATLAB scripts, functions and .mat files required to implement the method described in the manuscript:
C. Gouveia, D. Albuquerque, F. Barros, S. C. Soares, P. Pinho, J. Vieira, S. Bras, "Remote Emotion Recognition using Continuous-Wave Bio-Radar System", submitted to Behavior Research Methods, July 2023

In order to test the files, the signals of a single subject are available as an example.
Vital signs of the emotional conditions are available (bio-radar signal, ECG and respiratory signal of BIOPAC) to demonstrate how to extract features. Then, the observation matrix correspondent to the bio-radar signal and already without the mean value of the baseline is available, to demonstrate the statistical analysis and the classifiers implementation.
The MATLAB code was adapted to run for a single subject. The `Train_test_Classifiers.m` script shows also the process of choosing one subject to be left from the hold-out data split, serving as new data.

The files names have the following format: 'name_X', where 'X' is organized in the following work stages:
1. Extract features from bio-signals and create observation matrix;
2. Statistical study to select features;
3. Train and test classification models.

# Folders Description
* `Dataset_ID07_condition_1` - folder with signals from emotional condition, where `bR` is the bio-radar signal (containing respiratory and cardiac signal), `bP` is the respiratory signal from BIOPAC and ECG is the `ECG` from BIOPAC.
* `Matrix_wo_baseline_1` - folder with observation matrix without the mean value of baselines (exclusive for bio-radar signals).

# MATLAB Scripts Description
* `observation_matrix_1.m` - create the observation matrix, by extracting features over the bio-radar signals.
* `observation_matrix_BIOPAC_1.m` - create the observation matrix, by extracting features over the BIOPAC signals.
* `select_features_2.m`- statistical study to select features.
* `Train_test_Classifiers_3.m` - implements and tests the classifiers.

# MATLAB Functions Description
* `DFA_fun_1.m` - function to implement the DFA analysis, without any changes from (https://www.mathworks.com/matlabcentral/fileexchange/67889-detrended-fluctuation-analysis-dfa).
* `load_dataset_singleID_1.m` - load bio-radar signals per subject and create a signal matrix for all emotion conditions.
* `load_dataset_singleID_BIOPAC_1.m` - load BIOPAC signals per subject and create a signal matrix for all emotion conditions.
* `features_extraction_1.m` - extract features from bio-radar signals.
* `features_extraction_BIOPAC_1.m` - extract features from BIOPAC signals.
* `load_Omatrix_2.m` - load the observation matrix and assigns labels according to the desired test condition.
* `generate_class_3.m` - generates labels according to the desired test condition.
* `init_acc_3.m` - initializes vectors to store model performance results.

# MAT Files Description
* `ANN_HR_matrix_ID7_1.mat` - observation matrix to be used in the ANN model to determine the bio-radar heart rate with a lower error.
* `ANN_HR_Mdl_ID7_1.mat` - individual ANN model to determine the bio-radar heart rate with a lower error.
* `filterFIR1_1.mat` - Band-pass filter coefficients. This band-pass filter was applied to the bio-radar respiratory signal, before extracting the cardiac signal with wavelets.
* `selected_features_2.mat` - vector with the features selected in the statistical study.
* `matrix_ID_7F_1.mat`, `matrix_ID_7H_1.mat`, `matrix_ID_7N_1.mat`, `matrix_ID_BP_7F_1.mat`, `matrix_ID_BP_7H_1.mat`, `matrix_ID_BP_7N_1.mat`- observation matrix created with scripts `observation_matrix_.m` and `observation_matrix_BIOPAC_1.m`.