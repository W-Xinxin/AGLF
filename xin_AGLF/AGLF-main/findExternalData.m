function [X] = findExternalData(data, nvList, index) % modified from findindex.m 

%FINDINDEX Summary of this function goes here
% 目的： 对于当前视图中缺失的样本， 找到其他视图对应位置的数据
% data :    original data with 0 padding for missing view
% nvListL:  external views
% index:    the index for missing view

nv = length(nvList);
if nv == 1;
   X1 = data{nvList};
   X = X1(:,index);
   return; 
end

n1 = nvList(1);
tempX = data{n1};
for iv = 2: length(nvList)
    num = nvList(iv);
    tempX =  cat(1,tempX,data{num});
end
X = tempX(:,index);

end