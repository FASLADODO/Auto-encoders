[xTrainImages,tTrain] = digitTrainCellArrayData;
x1=[28,28]
for i = 1:numel(xTrainImages)
    
xTrainImages{i}=rgb2gray(xTrainImages{i});
xTrainImages{i}=imresize(xTrainImages{i},x1);
    
    
end
rng('default')
hiddenSize1 = 200;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',500, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
feat1 = encode(autoenc1,xTrainImages);

hiddenSize2 = 150;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',500, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);
hiddenSize3 = 100;
autoenc3 = trainAutoencoder(feat2,hiddenSize3, ...
    'MaxEpochs',500, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat3 = encode(autoenc3,feat2);
hiddenSize4 = 50;
autoenc4 = trainAutoencoder(feat3,hiddenSize4, ...
    'MaxEpochs',500, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat4 = encode(autoenc4,feat3);
hiddenSize5= 25;
autoenc5 = trainAutoencoder(feat4,hiddenSize5, ...
    'MaxEpochs',500, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

feat5 = encode(autoenc5,feat4);



softnet = trainSoftmaxLayer(feat5,tTrain,'MaxEpochs',500);

% view(softnet)
% view(autoenc1)
% view(autoenc2)
% view(softnet)
deepnet = stack(autoenc1,autoenc2,autoenc3,autoenc4,autoenc5,softnet);

% view(deepnet)

% Get the number of pixels in each image
imageWidth = x1(1);
imageHeight = x1(2);
inputSize = imageWidth*imageHeight;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
[xTestImages,tTest] = digitTestCellArrayData;

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
for j = 1:numel( xTestImages)
     xTestImages{j}=rgb2gray(xTestImages{j});
    xTestImages{j}=imresize(xTestImages{j},x1);
end
for i = 1:numel( xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end

y = deepnet(xTest);
plotconfusion(tTest,y);

% 
% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(xTrainImages));

for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
    
end

% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);

y = deepnet(xTest);
plotconfusion(tTest,y);
