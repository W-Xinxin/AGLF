function [X1,X2, ind] = findExistIndex(data, index) % modified from findindex.m 

%FINDINDEX Summary of this function goes here
%  Detailed explanation goes here
%  index : existed  view 

[numofview,~] = size(data);
[~,numofsample] = size(data{1});

X1 = cell(numofview,1);
X2 = cell(numofview,1);

ind = zeros(numofsample,numofview);
for i=1:numofview
    [d,~]=size(data{i});
    ind(index{i}, i) = 1;
    origin = data{i};
    origin(isnan(origin)) = 0;       % replace NULL with 0;
    X1{i} = NormalizeData(origin);  
    
    MissIndex{i} = find(1-ind(:,i));  % find missing index
    origin(:,MissIndex{i}) = [];      % omit the missing view
    X2{i} = NormalizeData(origin);  
end

end

