%%  Cross-view Anchor Graph Learning and Factorization for Incomplete Multi-view Clustering
%% min SUM_v=1^m av^2 |Ev + MvPv -GvFv|_F^2 + lambda1 |F|_sp^p + lambda2 |W_!vX_!v-A_!vMv|_F^2
%% s.t. Mv'1=1, Mv >=0, Gv'Gv =I, Fv>=0, W_!v'W_!v =I, A_!v'A_!v = Im
%%  22/06/2026 

clear;
clc;
addpath(genpath('./'));

resultdir1 = 'Results/';
if (~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

resultdir2 = 'aResults/';
if (~exist('aResults', 'file'))
    mkdir('aResults');
    addpath(genpath('aResults/'));
end

datadir='./datasets/';

dataname={'BDGP_fea','Caltech101-7','NUSWIDEOBJ','Animal','Wiki_fea','Caltech101-20','fmnist','MSRCv1','CCV'};

numdata = length(dataname); % number of the test datasets
numname = {'_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4','_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9'};

for idata =1
    ResBest = zeros(9, 8);
    ResStd = zeros(9, 8);
    for dataIndex = 1
        datafile = [datadir, cell2mat(dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        load(datafile);
        %data preparation...
        gt = truelabel{1};
        cls_num = length(unique(gt));
        k= cls_num;
        tic;
        [X1,Xe,ind] = findExistIndex(data, index); %% replace NULL with 0;  coded by xinxin
           
        time1 = toc;
        maxAcc = 0;
        TempAnchor = [1*k,3*k,5*k,7*k,10*k]; % for anchor graph  50*k
%         TempAnchor = [20*k]; % for anchor graph  50*k

        TempLambda1= [1e-2,1e-1,1,1e1,1e2];
%         TempLambda1= [0.01];

        TempLambda2 =[1e-5,1e-3,1e-1,1,1e1,1e3,1e5]; 
%         TempLambda2 =[1e-5]; 

        TempP = [0.1:0.1:1];  % for low rank tensor 
%         TempP = [0.1];  % for low rank tensor 

        
        ACC = zeros(length(TempAnchor),length(TempLambda1),length(TempLambda2),length(TempP));
        NMI = zeros(length(TempAnchor),length(TempLambda1),length(TempLambda2),length(TempP));
        Purity = zeros(length(TempAnchor),length(TempLambda1),length(TempLambda2),length(TempP));
        idx = 1;
          for anchorIndex =1: length(TempAnchor )
             numA = TempAnchor(anchorIndex); 
            for LambdaIndex1 = 1 : length(TempLambda1)
             lambda1 = TempLambda1(LambdaIndex1);  
             for LambdaIndex2 = 1 : length(TempLambda2) 
              lambda2 = TempLambda2(LambdaIndex2);  
              for pIndex3 = 1 : length(TempP)
                p = TempP(pIndex3);
                disp([char(dataname(idata)), char(numname(dataIndex)),'-an=', num2str(numA),'-l1=', num2str(lambda1),'-b2=', num2str(lambda2) '-p=', num2str(p)]);
                tic;
                
               %% construct anchor graph  
               %% method 1: k-means
                rand('seed',6666);                
                for iv =1 : length(Xe)
                   [~,anchor{iv}] = litekmeans(Xe{iv}',numA, 'MaxIter', 100,'Replicates',10); % m *dv
                    B{iv} = ConstructA_NP(Xe{iv}, anchor{iv}');  %% EMKMC, Yangben TIP TKDE2022
                end
                 B1=  GraphPadding(B',ind);     % padding anchor  graph with 0     
                 method = 101;

                para.c = cls_num; % K: number of clusters
                [pho_rho,PreY,obj] = algo_AGLF(B1,X1,gt,lambda1, lambda2,p,ind); % X,Y,lambda,d,numanchor
                time2 = toc;
                
                tic;
                for rep = 1 : 10
                    res(rep, : ) = Clustering8Measure(gt, PreY);
                end
                time3 = toc;

                runtime(idx) = time1 + time2 + time3/10; 
                disp(['runtime:', num2str(runtime(idx))])
                idx = idx + 1;
                tempResBest(dataIndex, : ) = mean(res);
                tempResStd(dataIndex, : ) = std(res);
                ACC(anchorIndex,LambdaIndex1, LambdaIndex2,pIndex3) = tempResBest(dataIndex, 1);
                NMI(anchorIndex,LambdaIndex1, LambdaIndex2,pIndex3) = tempResBest(dataIndex, 2);
                Purity(anchorIndex,LambdaIndex1, LambdaIndex2,pIndex3) = tempResBest(dataIndex, 3);
                save([resultdir1, char(dataname(idata)), char(numname(dataIndex)), '-an=', num2str(numA), '-l1=', num2str(lambda1),'-b2=', num2str(lambda2),'-p=', num2str(p), ...
                    '-acc=', num2str(tempResBest(dataIndex,1)), '_result.mat'], 'tempResBest', 'tempResStd');
                for tempIndex = 1 : 8
                    if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                        ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                        ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                    end
                end
               end
              end
            end
          end
        aRuntime = mean(runtime);
        PResBest = ResBest(dataIndex, :);
        PResStd = ResStd(dataIndex, :);
        save([resultdir2, char(dataname(idata)), char(numname(dataIndex)),char('m2_'), 'ACC_', num2str(max(ACC(:))), '_result.mat'], 'ACC', 'NMI', 'Purity', 'aRuntime', ...
            'PResBest', 'PResStd','TempAnchor','TempLambda1','TempLambda2','TempP','method','pho_rho');
    end
    save([resultdir2, char(dataname(idata)), '_result.mat'], 'ResBest', 'ResStd');
end
