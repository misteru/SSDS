function [Z, score, index] = SSDS(X, c, alpha, beta, gamma, NITER, group_num, sigma)
%-------------------------------------------------------------------------------------------
%Reference to be updated: "Symmetrical Self-representation and Data-grouping Strategy for Unsupervised Feature Selection"
%-------------------------------------------------------------------------------------------
%Input:
%      X: d by n matrix, n samples with d dimensions.
%      c: the desired cluster number.
%      alpha, beta, lambda, gamma, sigma: hyper-parameters, refer to the paper for details.
%      NITER: the desired number of iteration.
%      group_num: the number of groups
%Output:
%      Z: d by c projection matrix used for feature selection.
%      score: d-dimensional vector, preserves the score for each dimensions.
%      indx: the sort of features for selection.
%-------------------------------------------------------------------------------------------

%% Declare Variables
[d, n] = size(X);
Yp = orth(randn(n, c));
Z = randn(d, c);
S = abs(randn(n, n)); 
%S = ones(n, n) / n;
W = abs(randn(1, group_num));W = W / sum(W);
LapMatrix = randn(n, n);
 
%% Initialize similarity matrix for each view
group_label = zeros(d,1);
for i = group_num:-1:1
    group_label(1:floor(d / group_num * i),1) = group_num - i + 1;
end

S_temp = S;
for group_i = 1:group_num
    indx = group_label==group_i;
    data_group_i = X(indx, :);
    for i = 1:n
        for j = 1:n
            S_temp(i,j) = exp(-(sum((data_group_i(:, i) - data_group_i(:, j)).^2))/sigma);
        end
        S_temp(i,:) = S_temp(i,:) ./ sum(S_temp(i,:));
    end
    S_group.data{group_i}=(S_temp + S_temp') ./ 2;
end
clear group_label S_temp indx data_v;
 
%% Iterative optimization
err = 1;
iter = 1;
obj = zeros(NITER, 5);
 
X_times_Xt = X * X';
 
while (err > 1e-10 && iter <= NITER)
    tic;
    %% Update Z
    D_weight = diag( 0.5./sqrt(sum(Z.*Z,2)+eps));
    Z = pinv(alpha * X_times_Xt + (1-alpha) * eye(d) + beta * D_weight) * X * Yp;
  
    %% Update Yp
    Yp = gpi(gamma * LapMatrix, X' * Z);
    
    %% Update S
    mu = zeros(1,n);
    c_ij_s = zeros(n,n);
    b_ij_s = zeros(n,n);
    temp_domi_s = sum(W .* W);
    % assign aij & bij fist
    for i = 1:n
        for j = 1:n
            
            for group_i = 1:group_num
                b_ij_s(i,j) = b_ij_s(i,j) + W(group_i)^2 * log(S_group.data{group_i}(i,j));
            end
            b_ij_s(i,j) = b_ij_s(i,j) / temp_domi_s;
            c_ij_s(i,j) = (gamma * 0.5 * norm(Yp(i,:) - Yp(j,:))^2) / temp_domi_s;
            S(i,j) = exp(1+b_ij_s(i,j)+c_ij_s(i,j));
        end
        S(i,:) = S(i,:) / sum(S(i,:));
    end
    S = (S + S') ./ 2;
    LapMatrix = diag(sum(S, 1)) - S;
    
    %% Update W: (a column vector)
    Dkl_sv_s = W;
    for group_i = 1:group_num
        Dkl_sv_s(group_i) = 0;
        for i = 1:n
            for j = 1:n
                Dkl_sv_s(group_i) = Dkl_sv_s(group_i) + S_group.data{group_i}(i,j) * log((S_group.data{group_i}(i,j)+eps) / max(S(i,j), eps));
            end
        end
        W(group_i) = 1 / Dkl_sv_s(group_i);
    end
    W = W / sum(W);
    
    %% Objective function value
    sum_Dkl = 0;
    for group_i = 1:group_num
        sum_Dkl = sum_Dkl + W(group_i)^2 * Dkl_sv_s(group_i);
    end
    
    obj(iter,1) = alpha * norm(X'*Z-Yp,'fro')^2;
    obj(iter,2) = (1 - alpha) * norm(X-Z*Yp','fro')^2;
    obj(iter,3) = beta * sum(sqrt(sum(Z.^2, 2)));
    obj(iter,4) = gamma * trace(Yp' * LapMatrix * Yp);
    obj(iter,5) = sum_Dkl;%lambda * sum_Dkl;
    
    t_obj = toc;
    fprintf('i=%d, time=%.2fs, obj=%4.2f\n', iter, t_obj, sum(obj(iter,:)));
    
    if iter > 1
        err = abs(obj(iter - 1) - obj(iter));
    end
    
    iter = iter + 1;
end
 
score = sum((Z .* Z), 2);
[~, index] = sort(score, 'descend');
end
