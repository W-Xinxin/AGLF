%%  Cross-view Anchor Graph Learning and Factorization for Incomplete Multi-view Clustering
%%  min SUM_v=1^m av^2 |Ev + MvPv -GvFv'|_F^2 + lambda |F|_sp^p + beta |W_!vX_!v-A_!vMv|_F^2
%% s.t. Mv'1=1, Mv >=0, Gv'Gv =I, Fv>=0, W_!v'W_!v =I, A_!v'A_!v = Im
%%  22/06/2026 

function [pho_rho,Index,obj] = algo_AGLF(E, X, label ,lambda, beta,p ,ind)
% labels : ground truth   n *1.
% lambda1/2 : the hyper-parameter.
% p      :  the parameter of tensor learning
% ind    £º missing index : n *view
% m      :  anchor number
% X      :  incomplete original data with 0 padding 

% Ev     : m * n    anchor graph on existed view;   with zero pedding
% Mv     : m * nv   missing anchor graph; 
% Pv     : nv * n   index matrix 
% Gv     : m * k    orthogonal basis
% Fv     : k * n    non-negative label matrix
% W_!v   : p ^ d    orthogonal projection
% X_!v   : d * nv   cross-view data
% A_!v   : p * m    anchor matirx 

nV = length(X); N = size(X{1},2);
k = length(unique(label));
weight_vector = ones(1,nV)';        % the defult weight_vector of tensor Schatten p-norm
m = size(E{1},2);

%% ==============Variable Initialization=========%%
nvlist = (1:nV);
Xp = cell(1,nV);  

for iv = 1:nV
    E{iv} = E{iv}';
    extralViews = nvlist;
    extralViews(iv) = [];
    MissIndex{iv} = find(1-ind(:,iv));
    Xp{iv} =  findExternalData(X,extralViews,MissIndex{iv});    
    dim = size(Xp{iv},1);
    
    M{iv} = zeros(m,length(MissIndex{iv}));  
    W{iv} = zeros(m, dim);    % projection
    A{iv} = eye(m,m);
    
    Y{iv} = zeros(N, k);       % lagrange
    J{iv} = zeros(N, k);       % auxiliary variable

    G{iv} = zeros(m, k);
    F{iv} = zeros(N, k);
    
    Q{iv} = zeros(N, k);   % solve F
    QQ{iv}= zeros(N, k);   % solve J
    PQ{iv}= zeros(N,k);    % solve P
end
alpha = ones(1,nV)/nV;

%%
% disp('--------------Anchor Selection and Bipartite graph Construction----------');
% tic;
% opt1. style = 1;
% opt1. IterMax =50;
% opt1. toy = 0;
% [~, B] = My_Bipartite_Con(X,cls_num,0.5, opt1,10);
% toc;

%% =====================  Initialization =====================  
sX = [N, k, nV];
Isconverg = 0; iter = 1;
rho = 1e-4; max_rho = 10e12; pho_rho = 1.1;   % fault:1.1   penalty factor.;  upper bound,; update step
Pstops = 10e-3;

%% =====================Optimization=====================
while(Isconverg == 0)
    %% solve G{v}   
     for iv =1:nV
         fullM = zeros(m,N);
         fullM(:,MissIndex{iv}) = M{iv};
         QG{iv} = E{iv} + fullM;
         part1 = QG{iv}  * F{iv};
         [Unew,~,Vnew] = svd(part1,'econ');
         G{iv} = Unew*Vnew';
    end
    %% solve F{v}
    for iv =1:nV
        Q{iv} = (J{iv} - Y{iv}/rho);  %% 
        tempF = (alpha(iv)^2 * QG{iv}'* G{iv} + rho/2 * Q{iv})./(alpha(iv)^2 + rho/2);
        F{iv} = max(tempF,0);
    end
    
    %%  solve J{v}
    for iv =1:nV
        QQ{iv}=(F{iv} + Y{iv}/rho);
    end
    Q_tensor = cat(3,QQ{:,:});
    Qg = Q_tensor(:);
    [myj, ~] = wshrinkObj_weight_lp(Qg, lambda * weight_vector./rho,sX, 0,3,p);
    J_tensor = reshape(myj, sX);
    for k=1:nV
        J{k} = J_tensor(:,:,k);
    end
    
    %% solve M{v}
    for iv =1:nV
        Qm{iv} = E{iv} - G{iv}*F{iv}';
        tempM  = beta* A{iv}'*W{iv}*Xp{iv} - alpha(iv)^2* Qm{iv}(:,MissIndex{iv});
        for ii= 1: length(MissIndex{iv})
            ut = tempM(:,ii);
            M{iv}(:,ii) = EProjSimplex_new(ut');  
         end
    end
    
    %% solve A{v}   
    for iv =1: nV
       partA = W{iv}* Xp{iv}* M{iv}';
       [Unew,~,Vnew] = svd(partA,'econ');
       A{iv} = Unew*Vnew';
    end

    %% solve W{v}
    for iv =1: nV
       partW = A{iv}* M{iv}* Xp{iv}';
       [Unew,~,Vnew] = svd(partW,'econ');
       W{iv} = Unew*Vnew';
    end 
      
    %% solve av
    Ma = zeros(nV,1);
    for iv = 1:nV
        fullM = zeros(m,N);
        fullM(:,MissIndex{iv}) = M{iv};
        Ma(iv) = norm(Qm{iv}+ fullM ,'fro')^2 ;
    end
    Mfra = Ma.^-1;
    Qa = 1/sum(Mfra);
    alpha = Qa*Mfra;
    
    %% solve Y and  penalty parameters
    for iv=1:nV
        Y{iv} = Y{iv} + rho*(F{iv}-J{iv});
    end
    rho = min(rho*pho_rho, max_rho);

    %% compute loss value
    term1 = 0;
    term2 = 0;
    for iv = 1:nV
        term1 = term1 + alpha(iv)^2 * Ma(iv);
        term2 = term2 + beta * norm(W{iv}*Xp{iv}-A{iv}*M{iv} ,'fro')^2;
        res(iv) = norm(F{iv}-J{iv}, inf );
    end
    obj(iter) = term1+term2;
    res_max(iter) = max(res);

    %% ==============Max Training Epoc==============%%
    if (iter>10 && ( res_max(iter)< Pstops)) || iter >100   %% need iter> 100 for fmnist
    % if (iter>10 && ( res_max(iter)< Pstops))  %% need iter> 100 for fmnist
        Isconverg  = 1;
        SumF = 0;
        Sa = 0;
        for iv = 1: nV
            SumF = SumF + alpha(iv)^2 *F{iv};
            Sa = Sa + alpha(iv)^2;
        end
        SumF = SumF/Sa;
       [~,Index] = max(SumF,[],2);
       res = Clustering8Measure(label, Index); %%result = [ACC nmi Purity Fscore Precision Recall AR Entropy];
       fprintf('p :%d, ACC: %f,NMI: %f, Purity:%f,iter:%d  ',p , res(1),res(2),res(3),iter);   
    end
    iter = iter + 1;
end
