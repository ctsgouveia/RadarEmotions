function [Obsv_matrix] = features_extraction_BIOPAC_1(min_N,BP,ECG,fs)
%% Compute features
%% Input
% - min_N - Number of minutes per class
% - BP - Respiratory BIOPAC signal
% - fs - sampling frequency
%% Output
% - Obsv_matrix - observation matrix (features x result per minute)
%% FEATURES
% F1 - Heart rate (Hz)
% F46, F47 - SDNN for IBI-CS and IBI-RS
% F48, F49 - RMSSD for IBI-CS and IBI-RS
% F50 - pNN50 for IBI-CS
% F3, F4 - AppEn for CS and RS
% F15, F16, F17, F18 - Sk, Med, IQR, Av for IBI-CS
% F19, F20, F21, F22 - Sk, Med, IQR, Av for IBI-RS
% F23, F24, F25, F26 - Sk, Med, IQR, Av for CS
% F27, F28, F29, F30 - Sk, Med, IQR, Av for RS
% F31, F32, F33, F34 - Sk, Med, IQR, Av for inhale time (RS)
% F35, F36, F37, F38 - Sk, Med, IQR, Av for exhale time (RS)
% F5, F6 - mean absolute value of the first derivative of RS and RS-N
% F8, F9 - mean absolute value of the second derivative of RS and RS-N
% F7, F10 - mean absolute value of the first and second derivative of IBI-CS
% F51, F52 - DFA2 alpha1 and DFA2 alpha2 using IBI-CS
% F39, F44 - PSD in the bands: 0- 0.1Hz, 0.1 - 0.2Hz, 0.2 - 0.3Hz, 0.3 - 0.4Hz, 0.4 - 0.9Hz, 0.9 - 1.5Hz
% F45 - PSD Ratio LF/HF where LF corresponds to 0.1 - 0.4Hz and HF corresponds to 0.5 - 1.5Hz
% F11 - Energy ratio between the RS and the signal resulting from the Pre-BPF application
% F12 - Kurtosis RS
% F2 - Respiratory rate computed using zero-crossing peaks
% F13 - Mean Peak width for RS 
% F14 - Variance for RS
% F53, F54, F55, F56 - SD1, SD2, SD12 e SDRR for m = 1 (Poincaré plot)
% F57, F58, F59, F60 - SD1, SD2, SD12 e SDRR for m = 10 (Poincaré plot)

%% Initialize observation matrix
Obsv_matrix = zeros(60,min_N);

%% Compute HRV features using the sliding window method described in Fig.6
[SDNN_ibi_CS,RMSSD_ibi_CS,pNN50_ibi_CS,DFA2_alpha1,DFA2_alpha2,...
    SD1, SD2, SD12, SDRR, SD1_10, SD2_10, SD12_10, SDRR_10] = HRV_extract(ECG,fs,min_N);

%% Compute the remain features per minute
for k = 1:min_N
    % Open RS
    bP_RS = BP(k,:);
    % Open and normalizes CS
    bP_CS = ECG(k,:);
    bP_CS = bP_CS/max(bP_CS);
    
    % Compute IBI-CS
    dbg_ecg = 0;        % Activate debug mode
    th = 0.3;           % Threshold defined to find peaks
    ibi_CS = ibi_ecg(bP_CS,dbg_ecg,th,fs);

    % Heart rate
    med_ibi_ECG = median(ibi_CS);
    HR = 1/med_ibi_ECG;

    % Compute RS features
    [freq_RS,med_ibi_RS,peak_valey_width,ibi_RS,T_inhale,T_exhale] = freq_resp_ZC(bP_RS,fs);
    SDNN_RS = std(ibi_RS)*1000;
    dif_Win_RS = diff(ibi_RS);
    RMSSD_RS = sqrt(mean(dif_Win_RS.^2))*1000;

    % AppEn for RS and CS
    AppEn_resp = approximateEntropy(bP_RS);
    AppEn_heart = approximateEntropy(ECG(k,:));
    
    % PSD for RS
    [p01, p02, p03, p04, p05, p06, pLF, pHF] = PSD_analysis(bP_RS,fs);

    % Energy Ratio
    E_RS = sum(abs(bP_RS).^2)/length(bP_RS); 
    E_CS = sum(abs(ECG(k,:)).^2)/length(ECG(k,:)); 
    E_R = E_CS/E_RS;
   
    % Derivative features
    RS_N = bP_RS/max(bP_RS);
    
    d1_RS = abs(diff(bP_RS));
    d2_RS = abs(diff(bP_RS,2));
    Nd1_RS = abs(diff(RS_N));
    Nd2_RS = abs(diff(RS_N,2));
    d1_ibi_CS = abs(diff(ibi_CS));
    d2_ibi_CS = abs(diff(ibi_CS,2));

    % Build matrix
    Obsv_matrix(1,k) = HR;                      % F1 - Heart rate
    Obsv_matrix(46,k) = log(SDNN_ibi_CS);       % F46 - SDNN IBI-CS
    Obsv_matrix(47,k) = log(SDNN_RS);           % F47 - SDNN IBI-RS
    Obsv_matrix(48,k) = log(RMSSD_ibi_CS);      % F48 - RMSSD IBI-CS
    Obsv_matrix(49,k) = log(RMSSD_RS);          % F49 - RMSSD IBI-RS
    Obsv_matrix(50,k) = pNN50_ibi_CS;           % F50 - pNN50 IBI-CS
    Obsv_matrix(3,k) = AppEn_heart;             % F3 - AppEn CS
    Obsv_matrix(4,k) = AppEn_resp;              % F4 - AppEn RS
    
    Obsv_matrix(15,k) = skewness(ibi_CS);       % F15 - Skewness IBI-CS
    Obsv_matrix(16,k) = median(ibi_CS);         % F16 - Med IBI-CS
    Obsv_matrix(17,k) = iqr(ibi_CS);            % F17 - IQR IBI-CS
    Obsv_matrix(18,k) = mean(ibi_CS);           % F18 - Av IBI-CS
    
    Obsv_matrix(19,k) = skewness(ibi_RS);       % F19 - Skewness IBI-RS
    Obsv_matrix(20,k) = med_ibi_RS;             % F20 - Med IBI-RS
    Obsv_matrix(21,k) = iqr(ibi_RS);            % F21 - IQR IBI-RS
    Obsv_matrix(22,k) = mean(ibi_RS);           % F22 - Av IBI-RS

    Obsv_matrix(23,k) = skewness(ECG(k,:));     % F23 - Skewness CS
    Obsv_matrix(24,k) = median(ECG(k,:));       % F24 - Med CS
    Obsv_matrix(25,k) = iqr(ECG(k,:));          % F25 - IQR CS
    Obsv_matrix(26,k) = mean(ECG(k,:));         % F26 - Av CS

    Obsv_matrix(27,k) = skewness(bP_RS);        % F27 - Skewness RS
    Obsv_matrix(28,k) = median(bP_RS);          % F28 - Med RS
    Obsv_matrix(29,k) = iqr(bP_RS);             % F29 - IQR RS
    Obsv_matrix(30,k) = mean(bP_RS);            % F30 - Av RS

    Obsv_matrix(31,k) = skewness(T_inhale);     % F31 - Skewness T inhale
    Obsv_matrix(32,k) = median(T_inhale);       % F32 - Med T inhale
    Obsv_matrix(33,k) = iqr(T_inhale);          % F33 - IQR T inhale
    Obsv_matrix(34,k) = mean(T_inhale);         % F34 - Av T inhale

    Obsv_matrix(35,k) = skewness(T_exhale);     % F35 - Skewness T exhale
    Obsv_matrix(36,k) = median(T_exhale);       % F36 - Mediana T exhale
    Obsv_matrix(37,k) = iqr(T_exhale);          % F37 - IQR T inhale
    Obsv_matrix(38,k) = mean(T_exhale);         % F38 - Av T exhale

    Obsv_matrix(5,k) = mean(d1_RS);             % F5 - first derivative of RS
    Obsv_matrix(6,k) = mean(Nd1_RS);            % F6 - first derivative of RS-N
    Obsv_matrix(8,k) = mean(d2_RS);             % F8 - second derivative of RS
    Obsv_matrix(9,k) = mean(Nd2_RS);            % F9 - second derivative of RS-N
    Obsv_matrix(7,k) = mean(d1_ibi_CS);         % F7 - first derivative of IBI-CS
    Obsv_matrix(10,k) = mean(d2_ibi_CS);        % F10 - second derivative of IBI-CS


    Obsv_matrix(51,k) = DFA2_alpha1;            % F51 - DFA2 alpha 1 IBI-CS
    Obsv_matrix(52,k) = DFA2_alpha2;            % F52 - DFA2 alpha 2 IBI-CS
    
    Obsv_matrix(39,k) = mean(p01);              % F39 - PSD 0 - 0.1 Hz
    Obsv_matrix(40,k) = mean(p02);              % F40 - PSD 0.1 - 0.2 Hz
    Obsv_matrix(41,k) = mean(p03);              % F41 - PSD 0.2 - 0.3 Hz
    Obsv_matrix(42,k) = mean(p04);              % F42 - PSD 0.3 - 0.4 Hz
    Obsv_matrix(43,k) = mean(p05);              % F43 - PSD 0.4 - 0.9 Hz
    Obsv_matrix(44,k) = mean(p06);              % F44 - PSD 0.9 - 1.5 Hz
    Obsv_matrix(45,k) = mean(pLF)/mean(pHF);    % F45 - PSD LF/HF
    
    Obsv_matrix(11,k) = E_R;                    % F11 - Energy ratio RS
    Obsv_matrix(12,k) = kurtosis(bP_RS);        % F12 - Kurtosis RS
    Obsv_matrix(2,k) = freq_RS;                 % F2 - RS rate
    Obsv_matrix(13,k) = mean(peak_valey_width); % F13 - Peak width RS
    Obsv_matrix(14,k) = var(bP_RS);             % F14 - Variance RS
    
    Obsv_matrix(53,k) = SD1;                    % F53 - Poincare SD1, m = 1
    Obsv_matrix(54,k) = SD2;                    % F54 - Poincare SD2, m = 1
    Obsv_matrix(55,k) = SD12;                   % F55 - Poincare SD12, m = 1
    Obsv_matrix(56,k) = SDRR;                   % F56 - Poincare SDRR, m = 1

    Obsv_matrix(57,k) = SD1_10;                 % F57 - Poincare SD1, m = 10
    Obsv_matrix(58,k) = SD2_10;                 % F58 - Poincare SD2, m = 10
    Obsv_matrix(59,k) = SD12_10;                % F59 - Poincare SD12, m = 10
    Obsv_matrix(60,k) = SDRR_10;                % F60 - Poincare SDRR, m = 10
end
end

%% AUXILIARY FUNCTIONS ------------------------------------------------------------
%% Compute HRV features using the sliding window method
function [SDNN_ibi_CS,RMSSD_ibi_CS,pNN50_ibi_CS,DFA2_alpha1,DFA2_alpha2,...
    SD1, SD2, SD12, SDRR, SD1_10, SD2_10, SD12_10, SDRR_10] = HRV_extract(ECG,fs,min_N)
%%Inputs
% - ECG - ECG signal from BIOPAC
% - fs - sampling frequency
% - min_N - Number of minutes per class
%%Outputs
% - SDNN_ibi_CS - F46 SDNN IBI-CS
% - RMSSD_ibi_CS - F48 RMSSD IBI-CS
% - pNN50_ibi_CS - F50 pNN50 IBI-CS
% - DFA2_alpha1 - F51
% - DFA2_alpha2 - F52
% - SD1 - F53 Poincare m = 1
% - SD2 - F54 Poincare m = 1
% - SD12 - F55 Poincare m = 1
% - SDRR - F56 Poincare m = 1
% - SD1_10 - F57 Poincare m = 10
% - SD2_10 - F58 Poincare m = 10
% - SD12_10 - F59 Poincare m = 10
% - SDRR_10 - F60 Poincare m = 10
%%
% Build continuous signal
signalECG = ECG(1,:);
signalECG = signalECG/max(signalECG);

for k = 2:min_N
    signalECG = [signalECG ECG(k,:)/max(ECG(k,:))];
end

% IBI-CS
th_i = 0.3;                                     % Threshold defined to find peaks
dbg_ecg = 0;                                    % Activate debug mode
ibi_CS = ibi_ecg(signalECG,dbg_ecg,th_i,fs);

% DFA Analysis
pts_alpha1 = 10:10:100;
pts_alpha2 = 20:10:60;
[A,F] = DFA_fun_1(ibi_CS,pts_alpha1,2);
DFA2_alpha1 = A(1);
[A,F] = DFA_fun_1(ibi_CS,pts_alpha2,2);
DFA2_alpha2 = A(1);

% IBI, SDNN e RMSSD %
SDNN_ibi_CS = std(ibi_CS)*1000;
dif_Win_R = diff(ibi_CS);
RMSSD_ibi_CS = sqrt(mean(dif_Win_R.^2))*1000;

N_IBI = length(ibi_CS);
diff_ibi_msec = abs(diff(ibi_CS))*1000;
cnt_50_IBI = 0;                            % Initialize counter
for iii = 1:length(diff_ibi_msec)
    if diff_ibi_msec(iii) >= 50
        cnt_50_IBI = cnt_50_IBI +1;
    end
end
pNN50_ibi_CS = (cnt_50_IBI/N_IBI)*100;

% Poincare Plot with m = 1
xn = ibi_CS;
xn(end)=[];
xn1 = ibi_CS;
xn1(1)=[];
M_med_ibi = mean(xn);
autocov_0 = mean((xn-M_med_ibi).*(xn-M_med_ibi));
autocov_1 = mean((xn-M_med_ibi).*(xn1-M_med_ibi));
SD1 = sqrt(autocov_0-autocov_1);
SD2 = sqrt(autocov_0+autocov_1);
SDRR = sqrt(mean(xn.^2)-mean(xn)^2);
SD12 = SD1/SD2;

% Poincare Plot with m = 10
xn_10 = ibi_CS;
xn_10(end-9:end)=[];
xn1_10 = ibi_CS;
xn1_10(1:10)=[];
M_med_ibi = mean(xn_10);
autocov_0 = mean((xn_10-M_med_ibi).*(xn_10-M_med_ibi));
autocov_1 = mean((xn_10-M_med_ibi).*(xn1_10-M_med_ibi));
SD1_10 = sqrt(autocov_0-autocov_1);
SD2_10 = sqrt(autocov_0+autocov_1);
SDRR_10 = sqrt(mean(xn_10.^2)-mean(xn_10)^2);
SD12_10 = SD1/SD2;
end

%% RS features with filtered RS to emphasize and identify zero crossings
function [freq_RS,med_ibi_RS,peak_valey_width,ibi_RS,T_inhale,T_exhale] = freq_resp_ZC(bP_RS,fs)
%%Inputs
% - bP_RS - BIOPAC respiratory signal signal
% - fs - sampling frequency
%%Outputs
% - freq_RS - F2 RS rate
% - med_ibi_RS - F20 Med IBI-RS
% - peak_valey_width - F13 Peak width RS
% - ibi_RS - ibi in seconds
% - T_inhale - vector time inhales
% - T_exhale - vector time exhales
%%
% Filter to smooth RS
B = fir1(200,0.5/(fs/2));
signal_F = fftfilt(B,[bP_RS zeros(1,100)]);
signal_F(1:100) = [];
bP_RS = signal_F;

% Remove signal mean and normalize
M = movmean(bP_RS,1000);
bP_RS =  bP_RS - M;
bP_RS = bP_RS/max(bP_RS);

t_seg = (0:length(bP_RS)-1)*(1/fs);         % time vector
dbg = 0;                                    % debug mode off
thresh = 1;                                 % flag to apply temporal threshold
flg_CS = 0;                                 % flag acknowledge the usage in cardiac signal
[peak_valey_width,ibi_RS,T_inhale,T_exhale] = ibi_radar_F(t_seg,bP_RS,dbg,thresh,flg_CS);
med_ibi_RS = median(ibi_RS);
freq_RS = 1/med_ibi_RS;
end

%% Zero crossing method to compute signal rate and IBI
function [peak_valey_width,ibi_RS,T_inhale,T_exhale] = ibi_radar_F(t_seg,seg,dbg,thresh,flg_CS)
%%Inputs
% - t_seg - time vector
% - seg - signal segment under evaluation
% - dbg - debug mode shows the detected peaks
% - thresh - flag to apply temporal threshold to avoid outlier peaks
% - flg_CS - flag acknowledge the usage in cardiac signal
%%Outputs
% - peak_valey_width - width of peaks and valleys (seconds)
% - ibi_RS - IBI in seconds
% - T_inhale - vector time inhales
% - T_exhale - vector time exhales
%%
% Define zero-crossing function
zci = @(v) find(v(:).*circshift(v(:), [-1 0]) <= 0);
zx = zci(seg);
LOCS_0 = t_seg(zx);                     % time location of zero-crossing
PKS_0 = seg(zx);                        % signal location of zero-crossing

peak_valey_width = diff(LOCS_0);        % width of peaks and valleys

% Debug mode: shows signal zero-crossings
if dbg
    figure(100);
    plot(t_seg, seg,'k'); hold on;
    hp = plot(LOCS_0, PKS_0, 'g*');
    set(hp,'LineWidth',1.5);
    pause;
end

% Detect signal peaks: computes the maximum value between odd zero-crossing pairs
LOCS_i = zeros(1,length(PKS_0)/2);      % Initialize peak time vector
PKS_i = zeros(1,length(PKS_0)/2);       % Initialize peak location vector
ii = 1;                                 % location index

for K = 1:2:length(PKS_0)-2
    aux = zx(K):zx(K+2);                % Select signal segment between odd zero-crossings
    [M,Index] = max(seg(aux));          % Computes the maximum of segment
    LOCS_i(ii) = t_seg(aux(Index));     % Stores time location of peak
    PKS_i(ii) = M;                      % Stores peak magnitude
    ii = ii+1;                          
end

% Includes the last peak
aux = zx(K+2):zx(K+3);                  
[M,Index] = max(seg(aux));              
LOCS_i(ii+1) = t_seg(aux(Index));       
PKS_i(ii+1) = M;

% Removes negative peaks
LOCS_i = LOCS_i(PKS_i > 0);
PKS_i = PKS_i(PKS_i > 0);

% Debug mode: shows signal zero-crossings
if dbg && ~thresh
    figure(100);
    hp1 = plot(LOCS_i, PKS_i, 'b*');
    set(hp1,'LineWidth',1.5);
    hold off;
    pause;
end

% Computes RS-IBI
ibi_RS = diff(LOCS_i);

% Applies time threshold to avoid outlier peaks
if thresh
   Thresh_t = mean(ibi_RS)/2;   % threshold is the mean IBI
   
   % Creates new location and peaks vector
    LOCS_i2 = 0;                % Initialize peak time vector
    PKS_i2 = 0;                 % Initialize peak location vector
    
    LOCS_i2(1) = LOCS_i(1);     % First position
    PKS_i2(1) = PKS_i(1);       % First position
    K = 2;
    
    % Stores peaks spaced at least Thresh_t
    for jj = 2:length(LOCS_i)
        if diff(LOCS_i(jj-1:jj)) >= Thresh_t
            LOCS_i2(K) = LOCS_i(jj);
            PKS_i2(K) = PKS_i(jj);
        end
        K = K+1;
    end
    
    % Removes negative peaks
    LOCS_i = LOCS_i2(PKS_i2 > 0.005);
    PKS_i = PKS_i2(PKS_i2 > 0.005);
    
     % Computes RS-IBI
    ibi_RS = diff(LOCS_i);

    % Debug mode: shows signal zero-crossings
    if dbg && thresh
        figure(100);
        hp1 = plot(LOCS_i, PKS_i, 'b*');
        set(hp1,'LineWidth',1.5);
        hold off;
        pause;
    end
end

% Compute RS features (if we are working with RS)
% Compute RS features (if we are working with RS)
if ~flg_CS
    % Initialize ihnale and exhale time vectors
    T_inhale = zeros(1,length(LOCS_i));
    T_exhale = zeros(1,length(LOCS_i));

    % Verifies if signal starts in peak or valley
    aux = zx(1):zx(2);                  % Signal segment between zero-crossing pair
    [M,Index] = max(seg(aux));          % Computes maximum of segment

    if M >= 0.05     % We start with a peak
        JJJ = 1;     % Inicialize zero-crossing index
    else             % We start with a valley
        JJJ = 2;     % Inicialize zero-crossing index
    end
    
    % Compute ihnale and exhale time vectors
    for jjj = 1:length(LOCS_i)
        T_inhale(jjj) = LOCS_i(jjj)-LOCS_0(JJJ);
        T_exhale(jjj) = LOCS_0(JJJ+1)-LOCS_i(jjj);
        JJJ = JJJ + 2;
        if JJJ >= length(LOCS_0)
            break;
        end
    end
else
    % For CS these vectors are zeros
    T_inhale = zeros(1,length(LOCS_i));
    T_exhale = zeros(1,length(LOCS_i));
end
end

%% PSD analysis
function [p01, p02, p03, p04, p05, p06, pLF, pHF] = PSD_analysis(bP_RS,fs)
%%Inputs
% - bP_RS - BIOPAC respiratory signal
% - fs - sampling frequency
%%Outputs
% - p01 : 0 - 0.1 Hz
% - p02 : 0.1 - 0.2 Hz
% - p03 : 0.2 - 0.3 Hz
% - p04 : 0.3 - 0.4 Hz
% - p05 : 0.4 - 0.9 Hz
% - p06 : 0.9 - 1.5 Hz
% - pLF : 0.1 - 0.4 Hz
% - pHF : 0.5 - 1.5 Hz
%%
% Remove mean and comput PSD
seg = bP_RS - mean(bP_RS);
[Pxx,Hz] = pwelch(seg,[],[],[],'power',fs);

[I,J6] = find(Hz' < 0.1);
p01 = zeros(1,length(J6));
for K6=1:length(J6)
    p01(K6)= Pxx(J6(K6));
end

[I,J7] = find(Hz' > 0.1 & Hz' < 0.2);
p02 = zeros(1,length(J7));
for K7=1:length(J7)
    p02(K7)= Pxx(J7(K7));
end

[I,J8] = find(Hz' > 0.2 & Hz' < 0.3);
p03 = zeros(1,length(J8));
for K8=1:length(J8)
    p03(K8)= Pxx(J8(K8));
end

[I,J9] = find(Hz' > 0.3 & Hz' < 0.4);
p04 = zeros(1,length(J9));
for K9=1:length(J9)
    p04(K9)= Pxx(J9(K9));
end

[I,J10] = find(Hz' > 0.4 & Hz' < 0.9);
p05 = zeros(1,length(J10));
for K10=1:length(J10)
    p05(K10)= Pxx(J10(K10));
end

[I,J11] = find(Hz' > 0.9 & Hz' < 1.5);
p06 = zeros(1,length(J11));
for K11=1:length(J11)
    p06(K11)= Pxx(J11(K11));
end

% LF
[I,JLF] = find(Hz' > 0.1 & Hz' < 0.5);
pLF = zeros(1,length(JLF));
for KLF=1:length(JLF)
    pLF(KLF)= Pxx(JLF(KLF));
end

% HF
[I,JHF] = find(Hz' > 0.6 & Hz' < 1.5);
pHF = zeros(1,length(JHF));
for KHF=1:length(JHF)
    pHF(KHF)= Pxx(JHF(KHF));
end
end

%% Compute IBI between R peaks of ECG
function [ibi_CS] = ibi_ecg(ecg,dbg_ecg,th,fs)
%%Inputs
% - ecg - ECG signal
% - dbg_ecg - debug mode shows the detected ECG peaks
% - fs - sampling frequency
%%Outputs
% - ibi_CS - IBI of ECG signal

%% Detect peaks with find peaks
t = (0:length(ecg)-1)*(1/fs);
[PKS,LOCS] = findpeaks(ecg, t, 'MinPeakHeight', th, 'MinPeakDistance',0.4);
% Debug mode
if dbg_ecg
    figure(100); plot(t, ecg); hold on;
    plot(LOCS, PKS, 'r*'); hold off;
    pause;
end
%% Computes IBI_CS
ibi_CS = diff(LOCS);
end

