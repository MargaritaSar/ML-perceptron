% Load train & test images & labels
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
tstimages = loadMNISTImages('t10k-images.idx3-ubyte');
tstlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% Reshape and show f.e the first 200 digits
%figure
for i = 1:200
subplot(10,10,i)
digit = reshape(images(:, i), [28,28]);
imshow(digit)
title(num2str(labels(i)))
end

% Prepare the arguments for the NN
labels = labels';
% dummyvar function doesn't take zeroes
labels(labels==0)=10;
labels=dummyvar(labels)'; %
tstlabels = tstlabels';
tstlabels(tstlabels==0)=10;
tstlabels=dummyvar(tstlabels)';
%trainFcn = 'trainscg';

% use Gradient descent with adaptive learning rate
trainFcn = 'traingda';

%MLP using back-propagation algorithm

i = 50;
hiddenLayerSize = i;
net = patternnet(hiddenLayerSize, trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand'; % Divide data randomly
% If traingda is used
net.trainParam.lr = 0.01;
net.trainParam.epochs = 1000; %default number

% Choose a Performance Function
% Cross-Entropy
net.performFcn = 'crossentropy';
% Choose Plot Functions
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotconfusion'};

% Train the Network
[net,tr] = train(net,images,labels);
% Test the Network with the MNIST training data
y = net(images);
performance_netdata = perform(net,labels,y);
tind = vec2ind(labels);
yind = vec2ind(y);
percentErrors_netdata = sum(tind ~= yind)/numel(tind);
% Test the Network with MNIST test data
tsty = net(tstimages);
performance_testdata = perform(net,tstlabels,tsty);
tind = vec2ind(tstlabels);
yind = vec2ind(tsty);
percentErrors_testdata = sum(tind ~= yind)/numel(tind);