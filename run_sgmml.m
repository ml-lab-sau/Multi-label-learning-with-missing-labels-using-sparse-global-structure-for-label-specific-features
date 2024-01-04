%SGMML Sparse Global Structure for label specific features for multi-label
%learning with missing labels
clear
addpath(genpath('.'));
clc

%function [result]=readme_LSML(optmParameter,modelparameter,data)
optmParameter.lambda1   = 10^1;  %  10^-3 %  missing labels
optmParameter.lambda2   = 10^-5; %  
optmParameter.lambda3   = 10^-5; %  
optmParameter.lambda4   = 10^-5; %  
optmParameter.lambda5   = 10^-5; %  Sanjay
optmParameter.lambda6   = 10^-5; %  regularization of C 
optmParameter.eta       = 10^-2; %  regularization of second-orde
optmParameter.maxIter   = 15;
optmParameter.tuneParaOneTime   = 0;
%optmParameter.rho   = 8; %0.099; set inside loop.
    
%% Model Parametersf
modelparameter.cv_num             = 5;
modelparameter.repetitions        = 1;

model_SGMML.optmParameter = optmParameter;
model_SGMML.modelparameter = modelparameter;
model_SGMML.tuneThreshold = 0;% tune the threshold for mlc
fprintf('*** run SGMML for multi-label learning with missing labels ***\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load the dataset, you can download the other datasets from our website

%datasets={'CAL500.mat', 'genbase.mat', 'medical.mat', 'Image.mat'};
%rho={1, 1, 1, 1};
%datasets={'yeast', 'rcv1subset1_top944', 'rcv1subset2_top944', 'delicious', };
%rho={8, 8, 2, 16};

%misRate = {'0.5', '0.7', '0.9'};
%datasets={'CAL500.mat', 'birds.mat', 'genbase.mat', 'medical.mat', 'chess.mat', 'Image.mat', 'yeast.mat',  'rcv1subset1_top944.mat', 'rcv1subset2_top944.mat', 'rcv1subset3_top944.mat', 'chemistry.mat', 'cooking.mat' };
misRate = {0.6, 0.8};
%datasets = {'yeast.mat'};
datasets = {'foodtruck.mat'};
rho = {1, 1, 1, 1, 2, 1, 4, 8, 2, 2, 8, 4};


%For parameter sensitivity, will be used later
%lambda = {10^-9, 10^-8, 10^-7, 10^-6, 10^-5, 10^-4, 10^-3, 10^-2, 10^-1, 10^0, 10^1};

for mr=1:numel(misRate)
    model_SGMML.misRate = misRate{mr}; % missing rate of positive  class labels

for dc=1:numel(datasets)
    rng(42);
    load(datasets{dc});
    target = target'; %For foodtruck dataset only
    optmParameter.rho = rho{dc};
    
if exist('train_data','var')==1
    data    = [train_data;test_data];
    target  = [train_target,test_target];
end
clear train_data test_data train_target test_target

data      = double (data);
num_data  = size(data,1);
temp_data = data + eps;
temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
if sum(sum(isnan(temp_data)))>0
    temp_data = data+eps;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
end
temp_data = [temp_data,ones(num_data,1)];

%target(target == -1) = 0;
target(target == -1) = -1;

randorder = randperm(num_data);
cvResult  = zeros(16,modelparameter.cv_num);
%%
%target(target==-1)=0;
for i = 1:modelparameter.repetitions       
    for j = 1:modelparameter.cv_num
        fprintf('- Repetition - %d/%d,  Cross Validation - %d/%d', i, modelparameter.repetitions, j, modelparameter.cv_num);
        [cv_train_data,cv_train_target,cv_test_data,cv_test_target ] = generateCVSet( temp_data,target',randorder,j,modelparameter.cv_num );

        if model_SGMML.misRate > 0
             temptarget = cv_train_target;
             [IncompleteTarget, ~, ~, realpercent]= getIncompleteTarget(cv_train_target, model_SGMML.misRate,1); 
             fprintf('\n-- Missing rate:%.1f, Real Missing rate %.3f\n',model_SGMML.misRate, realpercent); 
        end
       %% Model Training
        modelLLE  = sgmml( cv_train_data, IncompleteTarget,optmParameter); 

        %% Prediction and evaluation
        Outputs = (cv_test_data*modelLLE.W)';
        if model_SGMML.tuneThreshold == 1
            fscore                 = (cv_train_data*modelLLE.W)';
            [ tau,  currentResult] = TuneThreshold( fscore, cv_train_target', 1, 2);
            Pre_Labels             = Predict(Outputs,tau);
        else
            %Pre_Labels = double(Outputs>=0.3);
            %Pre_Labels = double(Outputs>0.0);
            Pre_Labels = sign(Outputs);
        end
        fprintf('-- Evaluation\n');
        tmpResult = EvaluationAll(Pre_Labels,Outputs,cv_test_target');
        cvResult(:,j) = cvResult(:,j) + tmpResult;
        if 0
            numofFeatures = sum(modelLLE.W~=0);
            figure;
            bar(numofFeatures);
        end
    end
end
endtime = datestr(now,0);
cvResult = cvResult./modelparameter.repetitions;
Avg_Result      = zeros(16,2);
Avg_Result(:,1) = mean(cvResult,2);
Avg_Result(:,2) = std(cvResult,1,2);
model_SGMML.avgResult = Avg_Result;
model_SGMML.cvResult  = cvResult;
result =   Avg_Result;
PrintResults(Avg_Result);

filename='result-sgmml.xlsx';
resultToSave = Avg_Result([1, 6, 11:16], 1 );
xlColumn = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'};
xlLocation = [xlColumn{mr} num2str((8*(dc-1))+1)]; 
%xlLocation = [xlColumn{dc} num2str(1)]; %for parameter sensitivity with dc
%used for traversing over lambdas.
Sheet = 'lambda';
xlswrite(filename, resultToSave, Sheet, xlLocation);
end
end