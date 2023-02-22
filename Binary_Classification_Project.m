% Sanaz Hami Hassan Kiyadeh
clear
%Using Stats and machine learning toolbox, Computer vision toolbox
%Load the data
ImageFolder=fullfile('PetImages');

imds = imageDatastore(ImageFolder,'IncludeSubfolders',true,'LabelSource',...
    'foldernames');

% Split the data set into a training and test data. 
% Pick 30% of images from each set for the training data.
%[trainingSet,testSet] = splitEachLabel(imds,0.001,'randomize');
[trainingSet,testSet1,testSet2] = splitEachLabel(imds,0.001,0.001,'randomize');

bag = bagOfFeatures(trainingSet);

categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
ConfMatrix = evaluate(categoryClassifier,testSet1);
