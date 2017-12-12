function [beta] = SVMGD(X, y, lambda, T, eta)
% Using gradient descent this function computes 
% the objective argument          : beta
% the objective value             : f(beta)
% the objective error             : f(beta)-f(beta_hat)
% the optimization error          : norm(beta-beta_hat)
% and the total error             : norm(beta-beta_star).

% X        : design matrix;
% y        : response vector;
% lambda   : penalty parameter
% T        : number of iterations;
% eta      : step size in each iteration.
% beta_hat : optimal solution, calculated by LassoCVX
% beta_star: true parameter (stored as b in LassoData.mat)

    %%%%%%%%%%%%%%%%%% Initialization
    beta = zeros(size(X,2),1);
    %%%%%%%%%%%%%%%%%%

    % compute the optimal objective value
    %obj_hat = ;

    % Gradient Descent with fixed step size
    for t=1:T
        beta = beta - eta*df(beta, X, y, lambda);
    end
end

function df = df(beta, X, y, lambda)
    % This function computes the subgradient of the objective function at beta
    
    df = 0;
    n  = numel(y);
    
    for i = 1:n
        df = df + 100*dj(beta, X(i,:)', y(i));
    end
    
    df = df + lambda*dh(beta);
end

function dg = dg(x, y)
    dg = -y*x;
end

function dh = dh(beta)
    dh = vertcat(2*beta(1:end-1,1),0);
end

function dj = dj(beta, x, y)
    gval = g(beta, x, y);
    
    if gval == 0
        dj = 0;
    elseif gval > 0
        dj = dg(x, y);
    else
        dj = 0;
    end
end

function g = g(beta, x, y)
    g = 1-y*x.'*beta;
end

function h = h(beta)
    h = norm(beta);
end

function j = j(beta, x, y)
    gval = g(beta, x, y);
    
    if(gval > 0)
        j = gval;
    else
        j= 0;
    end    
end

function f = f(beta, X, y, lambda)
    f = 0;    
    n = numel(y);
    
    for i = 1:n
        y_i = y(i);
        x_i = X(i,:).';        
        
        f = f + 1/n * j(beta, x_i, y_i) ;
    end
    
    f = f + lambda * h(beta);
end

function k = k(x1,x2)
    k = x1'*x2;
end