function [X1, ind] = GraphPadding(graph, index)  % coded by xinxin

%FINDINDEX Summary of this function goes here
%  Detailed explanation goes here
%  index :  n * view;  1: exist, 0 : missing
%% 方法： 0初始化完整图， 在将图按索引放到存在的位置上。

[numofview,~] = size(graph);
[~,m]= size(graph{1});
[numofsample,~] = size(index);

G = cell(numofview,1);


for iv=1:numofview 
    gra = zeros(numofsample,m);            % initial anchor graph with 0
    ExistIndex{iv} = find(index(:,iv));    % find exist index
    gra(ExistIndex{iv},:) = graph{iv};     % omit the missing view
    X1{iv} = gra;  
    clear gra;
end




end

