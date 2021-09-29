%1. Load training set and testing set
clear all;
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');


% Implementation of K-Nearest Neighbors
% 1-NN

v_train = cat(1,v_train0,v_train1,v_train2,v_train3);
% Class of zeros
distances0 = zeros(980,24754); % Total number of training images = 24754
for i = 1:980
    for j = 1:24754
        distances0(i,j) = sqrt((v_train(j,1)-v_test0(i,1))^2+(v_train(j,2)-v_test0(i,2))^2);
    end
end
[M,I] = min(distances0,[],2);
count0 = 0;
% If the index isn't between the starting and ending point of the corresponding training
% set, the counter for the wrongly-classified images is augmented by one
for i = 1:980
    if I(i)>5923
        count0 = count0 + 1;
    end
end
p0 = count0/980;
% Class of ones
distances1 = zeros(1135,24754); 
for i = 1:1135
    for j = 1:24754
        distances1(i,j) = sqrt((v_train(j,1)-v_test1(i,1))^2+(v_train(j,2)-v_test1(i,2))^2);
    end
end
[M,I] = min(distances1,[],2);
count1 = 0;
for i = 1:1135
    if I(i)<=5923 || I(i)>12665
        count1 = count1 + 1;
    end
end
p1 = count1/1135;
% Class of twos
distances2 = zeros(1032,24754); 
for i = 1:1032
    for j = 1:24754
        distances2(i,j) = sqrt((v_train(j,1)-v_test2(i,1))^2+(v_train(j,2)-v_test2(i,2))^2);
    end
end
[M,I] = min(distances2,[],2);
count2 = 0;
for i = 1:1032
    if I(i)<=12665 || I(i)>18623
        count2 = count2 + 1;
    end
end
p2 = count2/1032;
% Class of threes
distances3 = zeros(1010,24754); 
for i = 1:1010
    for j = 1:24754
        distances3(i,j) = sqrt((v_train(j,1)-v_test3(i,1))^2+(v_train(j,2)-v_test3(i,2))^2);
    end
end
[M,I] = min(distances3,[],2);
count3 = 0;
for i = 1:1010
    if I(i)<=18623
        count3 = count3 + 1;
    end
end
p3 = count3/1010;

% The percentage of wrongly-classified images with 1-Nearest Neighbor
P_1NN = (p0+p1+p2+p3)/4; 

% 3-NN
% Class of zeros
% Sort the distances' matrix and find the most common class based on
% the first 3 values
[B,I] = sort(distances0,2);
count0 = 0;
count1 = 0;
count2 = 0;
count3 = 0;
C = zeros(980,1);
for i = 1:980
    for j = 1:3
        if I(i,j)<=5923
            count0 = count0 + 1;
        elseif I(i,j)<=12665
            count1 = count1 + 1;
        elseif I(i,j)<=18623
            count2 = count2 + 1;
        else
            count3 = count3 + 1;
        end
    end
    count = [count0, count1, count2, count3];
    a = max(count);
    if a == count0
        C(i)=0;
    elseif a == count1
        C(i)=1;
    elseif a == count2
        C(i)=2;
    else
        C(i)=3;
    end
end
% For each image that the most common class wasn't the same as
% the testing set's class, increase the counter for the wrongly-classified
% images by one
wrong = 0;
for i = 1:980
    if C(i) ~= 0
        wrong = wrong + 1;
    end
end
p0 = wrong/980;
% Class of ones
[B,I] = sort(distances1,2);
count0 = 0;
count1 = 0;
count2 = 0;
count3 = 0;
C = zeros(1135,1);
for i = 1:1135
    for j = 1:3
        if I(i,j)<=5923
            count0 = count0 + 1;
        elseif I(i,j)<=12665
            count1 = count1 + 1;
        elseif I(i,j)<=18623
            count2 = count2 + 1;
        else
            count3 = count3 + 1;
        end
    end
    count = [count0, count1, count2, count3];
    a = max(count);
    if a == count0
        C(i)=0;
    elseif a == count1
        C(i)=1;
    elseif a == count2
        C(i)=2;
    else
        C(i)=3;
    end
end
wrong = 0;
for i = 1:1135
    if C(i) ~= 1
        wrong = wrong + 1;
    end
end
p1 = wrong/1135;
% Class of twos
[B,I] = sort(distances2,2);
count0 = 0;
count1 = 0;
count2 = 0;
count3 = 0;
C = zeros(1032,1);
for i = 1:1032
    for j = 1:3
        if I(i,j)<=5923
            count0 = count0 + 1;
        elseif I(i,j)<=12665
            count1 = count1 + 1;
        elseif I(i,j)<=18623
            count2 = count2 + 1;
        else
            count3 = count3 + 1;
        end
    end
    count = [count0, count1, count2, count3];
    a = max(count);
    if a == count0
        C(i)=0;
    elseif a == count1
        C(i)=1;
    elseif a == count2
        C(i)=2;
    else
        C(i)=3;
    end
end
wrong = 0;
for i = 1:1032
    if C(i) ~= 2
        wrong = wrong + 1;
    end
end
p2 = wrong/1032;
% Class of threes
[B,I] = sort(distances3,2);
count0 = 0;
count1 = 0;
count2 = 0;
count3 = 0;
C = zeros(1010,1);
for i = 1:1010
    for j = 1:3
        if I(i,j)<=5923
            count0 = count0 + 1;
        elseif I(i,j)<=12665
            count1 = count1 + 1;
        elseif I(i,j)<=18623
            count2 = count2 + 1;
        else
            count3 = count3 + 1;
        end
    end
    count = [count0, count1, count2, count3];
    a = max(count);
    if a == count0
        C(i)=0;
    elseif a == count1
        C(i)=1;
    elseif a == count2
        C(i)=2;
    else
        C(i)=3;
    end
end
wrong = 0;
for i = 1:1010
    if C(i) ~= 3
        wrong = wrong + 1;
    end
end
p3 = wrong/1010;

% Total error of 3-NN
P_3NN = (p0+p1+p2+p3)/4;

% Implementation of Nearest Class Centroid

center0 = sum(v_train0)/5923;
center1 = sum(v_train1)/6742;
center2 = sum(v_train2)/5958;
center3 = sum(v_train3)/6131;
center = {center0,center1,center2,center3};
% Class of zeros
dist0 = zeros(980,4);
for i = 1:980
    for j = 1:4
        dist0(i,j) = sqrt((v_test0(i,1)-center{1,j}(1))^2 + (v_test0(i,2)-center{1,j}(2))^2);
    end
end
[M,I] = min(dist0,[],2);
% For each image that the minimum distance from the centers wasn't the one 
% corresponding to the testing set's class, increase the counter for the 
% wrongly-classified images by one
wrong0 = 0;
for i = 1:980
    if I(i)~=1
       wrong0 = wrong0 + 1;
    end
end
p0 = wrong0/980;
% Class of ones
dist1 = zeros(1135,4);
for i = 1:1135
    for j = 1:4
        dist1(i,j) = sqrt((v_test1(i,1)-center{1,j}(1))^2 + (v_test1(i,2)-center{1,j}(2))^2);
    end
end
[M,I] = min(dist1,[],2);
wrong1 = 0;
for i = 1:1135
    if I(i)~=2
       wrong1 = wrong1 + 1;
    end
end
p1 = wrong1/1135;
% Class of twos
dist2 = zeros(1032,4);
for i = 1:1032
    for j = 1:4
        dist2(i,j) = sqrt((v_test2(i,1)-center{1,j}(1))^2 + (v_test2(i,2)-center{1,j}(2))^2);
    end
end
[M,I] = min(dist2,[],2);
wrong2 = 0;
for i = 1:1032
    if I(i)~=3
       wrong2 = wrong2 + 1;
    end
end
p2 = wrong2/1032;
% Class of threes
dist3 = zeros(1010,4);
for i = 1:1010
    for j = 1:4
        dist3(i,j) = sqrt((v_test3(i,1)-center{1,j}(1))^2 + (v_test3(i,2)-center{1,j}(2))^2);
    end
end
[M,I] = min(dist3,[],2);
wrong3 = 0;
for i = 1:1010
    if I(i)~=4
       wrong3 = wrong3 + 1;
    end
end
p3 = wrong3/1010;

% Total error of Nearest Centroid Classifier
P_NC = (p0+p1+p2+p3)/4;

P_1NN , P_3NN, P_NC