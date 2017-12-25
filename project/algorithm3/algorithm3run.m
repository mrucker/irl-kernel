function irl_result = algorithm3run(algorithm_params, mdp_data, mdp_model, feature_data, example_samples, true_features, verbosity)

% algorithm_params - parameters of the MMP algorithm:
%       seed (0) - initialization for random seed
%       all_features (0) - use all features as a basis
%       true_features (0) - use true features as a basis
% mdp_data - definition of the MDP to be solved
% example_samples - cell array containing examples
% irl_result - result of IRL algorithm, generic and algorithm-specific:
%       r - inferred reward function
%       v - inferred value function.
%       q - corresponding q function.
%       p - corresponding policy.
%       time - total running time

    % Fill in default parameters.
    algorithm_params = algorithm3defaultparams(algorithm_params);

    % Set random seed.
    if algorithm_params.seed ~= 0
        rng(algorithm_params.seed);
    end

    tic;

    % Initialize variables.
    [states,actions,transitions] = size(mdp_data.sa_p);

    [N,T] = size(example_samples);            

    % Build feature membership matrix.
    if algorithm_params.all_features
        F = feature_data.splittable;
        F = horzcat(F,ones(states,1)); %We add a row of 1s to the feature matrix to ensure we can control the reward at every state                
    elseif algorithm_params.true_features
        F = true_features;
    else
        F = eye(states);
    end

    % Count features.
    
    F = F';

    features = size(F,1);

    % Construct state expectations.    
    nE = zeros(states,1);
    sE = zeros(states,1);
    
    for i=1:N
        for t=1:T
            nE(example_samples{i,t}(1)) = nE(example_samples{i,t}(1)) + 1;
            sE(example_samples{i,t}(1)) = sE(example_samples{i,t}(1)) + 1*mdp_data.discount^(t-1);
        end
    end
    
    nE = nE/N;
    sE = sE/N;
    
    draw(sE, nE);

    %true_r = algorithm_params.true_r;
    %true_p = standardmdpsolve(mdp_data, true_r);
    %sE     = standardmdpfrequency(mdp_data, true_p);
    mE     = F*sE;

    % Step 1
    rand_w = rand(features,1);
    rand_w = rand_w/norm(rand_w); %not important to the algorithm, but this allows for comparisons against future w's.
    rand_r = repmat(F'*rand_w, 1, actions);
    rand_p = standardmdpsolve(mdp_data, rand_r);
    rand_s = standardmdpfrequency(mdp_data, rand_p)/10;
    rand_m = F*rand_s;

    rs = {rand_r};
    ps = {rand_p};
    ss = {rand_s};
    ms = {rand_m};
    ts = {0};    

    i = 2;
    
    ff = k(F,F, algorithm_params.p, algorithm_params.s);
    
    while 1

        x = horzcat(sE, cell2mat(ss));        
        y = vertcat(1,-ones(i-1,1));
        
        %Step 2
        [m, g, b, u, r] = maxMarginOptimization_4_s(y, x, ff, verbosity);

        rd = r'*x(:,1) - r'*x(:,i);
        
        % Print t.
        if verbosity ~= 0
            fprintf(1,'Completed IRL iteration,i=%d, t=%f, g=%d, b=%d, u=%d, rd=%f\n',i,m,g,b,u,rd);
        end
                
        rs{i} = repmat(r, 1, actions);
        ts{i} = m;
        
        %Step 3
        if (i==200 || abs(ts{i}-ts{i-1}) <= algorithm_params.epsilon)% || ts{i} <= algorithm_params.epsilon || (i > 2 && ts{i}-ts{i-1} > ts{i-1}*2))
            break;
        end

        %Step 4        
        ps{i} = standardmdpsolve(mdp_data, rs{i});
        ss{i} = standardmdpfrequency(mdp_data, ps{i})/10;        
        ms{i}  = F*ss{i};

        %Step 6
        i = i+1;
    end
    
    % Compute mu for last policy    
    ps{i} = standardmdpsolve(mdp_data, rs{i});
    ss{i} = standardmdpfrequency(mdp_data, ps{i})/10;
    ms{i} = F*ss{i};
    
    idx = i;
    
    % In Abbeel & Ng's algorithm, we should use the weights lambda to construct
    % a stochastic policy. However, here we are evaluating IRL algorithms, so
    % we must return a single reward. To this end, we'll simply pick the reward
    % with the largest weight lambda.
    [~,idx] = max(mixPolicies_1(sE, ss, ff, verbosity));
    %[~,idx] = max(mixPolicies_2(mE, ms, ff, verbosity));

    time = toc;
    
    m = ts{idx};
    r = rs{idx};
    rd = r(:,1)'*x(:,1) - r(:,1)'*x(:,idx);
    
    if verbosity ~= 0
        fprintf(1,'Completed IRL iteration,i=%d, t=%f, g=%d, b=%d, u=%d, rd=%f\n',idx,m,0,0,i,rd);
    end
    
    %irl_result = marshallResults(rs{idx}, ws{idx}, mdp_model, mdp_data, time);
    irl_result = marshallResults(rs{idx}, 0, mdp_model, mdp_data, time);
end

function [margin, right, wrong, unknown, reward] = maxMarginOptimization_4_s(y, x, ff, verbosity, varargin)
    o_cnt = size(x,2);
    s_cnt = size(x,1);    

    vv = x'*ff*x;

    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables a(o_cnt) b1 r1(s_cnt);
        maximize(sum(a) - 1/2*quad_form(a.*y, vv)) %dual problem
        subject to
            %b1 == 1 - x(:,1)'*ff*x*(a.*y);
            %r1 == ff*x*(a.*y) + b1;
            0 == a'*y;
            0 <= a;
    cvx_end
    warning('off','all')
    
    sv = x(:,round(a,8)>0);
    sl = y(round(a,8)>0,1);
    
    %regarding b0: "we typically use an average of all the solutions for numerical stability" (ESL pg.421)
    b0 = mean(sl - sv'*ff*x*(a.*y)); %aka , -(a'*vv*(a.*y)/sum(a));
    rs = ff*x*(a.*y) + b0;
    
    %ds      = sign(y.*(vv*(a.*y) + b0));
    ds      = zeros(size(x,2),1);
    right   = sum(ds == 1);
    wrong   = sum(ds == -1);
    unknown = sum(ds == 0);
    reward  = rs;
    margin  = 1/sqrt(sum(a));    
end

function [margin, right, wrong, unknown, reward] = maxMarginOptimization_5_s(y, x, ff, verbosity, varargin)    
    o_cnt = size(x,2);
    s_cnt = size(x,1);    

    vv = x'*ff*x;

    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables a(o_cnt) b1 r1(s_cnt) zi(s_cnt);
        maximize(sum(a) - 1/2*quad_form(a.*y, vv)) %dual problem  - 1/50*sum(zi-min(zi))
        subject to
            %b1 == 1 - x(:,1)'*ff*x*(a.*y);
            %0 == ff*x*(a.*y) + b1 + zi;
            %0 <= zi;
            0 == a'*y;
            0 <= a;
    cvx_end
    warning('off','all')    
    
    %zi'-min(zi)
    
    sv = x(:,round(a,8)>0);
    sl = y(round(a,8)>0,1);
    
    %regarding b0: "we typically use an average of all the solutions for numerical stability" (ESL pg.421)
    b0 = mean(sl - sv'*ff*x*(a.*y)); %aka , -(a'*vv*(a.*y)/sum(a));
    rs = ff*x*(a.*y) + b0;
    
    %rs'
    
    %ds      = sign(y.*(vv*(a.*y) + b0));
    ds      = zeros(size(x,2),1);
    right   = sum(ds == 1);
    wrong   = sum(ds == -1);
    unknown = sum(ds == 0);
    reward  = rs;
    margin  = 1/sqrt(sum(a));    
end

function [lambda] = mixPolicies_1(sE, ss, ff, verbosity)
    s_mat = cell2mat(ss);
    
    f_cnt = size(s_mat,1);
    s_cnt = size(s_mat,2);

    ssffss = s_mat'*ff*s_mat;
    seffse = sE'*ff*sE;
    seffss = sE'*ff*s_mat;

    % Solve optimization to determine lambda weights.
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables l(s_cnt);
        minimize(l'*ssffss*l + seffse - 2*seffss*l);
        subject to
            l >= 0;
            1 == sum(l);
    cvx_end
    
    lambda = l;
end

function [lambda] = mixPolicies_2(mE, ms, ff, verbosity)
    m_mat = cell2mat(ms);
    
    f_cnt = size(m_mat,1);
    m_cnt = size(m_mat,2);
    
    % Solve optimization to determine lambda weights.
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables m(f_cnt) l(m_cnt);
        minimize(norm(m-mE));
        subject to
            m == m_mat*l;
            l >= 0;
            1 == sum(l);
    cvx_end
    
    lambda = l;
end

function irl_result = marshallResults(r, w, mdp_model, mdp_data, time)
    
    mdp_solve = str2func(strcat(mdp_model,'solve'));
    
    % Compute policies.
    soln = mdp_solve(mdp_data, r);    
    v = soln.v;
    q = soln.q;
    p = soln.p;

    r_itr      = cell(1,1);
    tree_r_itr = cell(1,1);
    p_itr      = cell(1,1);
    tree_p_itr = cell(1,1);
    wts_itr    = cell(1,1);

    r_itr{1}      = r;
    tree_r_itr{1} = r;
    p_itr{1}      = p;
    tree_p_itr{1} = p;
    wts_itr{1}    = w;

    % Construct returned structure.
    irl_result = struct('r',r,'v',v,'p',p,'q',q,'r_itr',{r_itr},'model_itr',{wts_itr},...
        'model_r_itr',{tree_r_itr},'p_itr',{p_itr},'model_p_itr',{tree_p_itr},...
        'time',time);
end

%Drawing
function draw(r1, r2)

n = sqrt(size(r1,1));

% Create final figure.
figure('Name','visitation comparison','NumberTitle','off');
hold on;
grid on;
cla;

% Draw reward for ground truth.
subplot(1,2,1);
title('Discounted Visits');
feval(strcat('gridworld','draw'), r1, [], [], struct('n',n));

% Draw reward for IRL result.
subplot(1,2,2);
title('True Visits');
feval(strcat('gridworld','draw'), r2, [], [], struct('n',n));

% Turn hold off.
hold off;
end
%Drawing

%Kernels
function m = parse(x1, x2, k)
    m = zeros(size(x1,2),size(x2,2));
    
    x1 = x1';
    
    for c = 1:size(x2,2)
        for r = 1:size(x1,1)
            m(r,c) = k(x1(r,:)', x2(:,c));
        end
    end
end

function m = k(x1, x2, varargin)
    p = varargin{1};
    s = varargin{2};
    
    %k1 = kernel_gaussian(s);
    %k2 = kernel_polynomial(p);
    %k3 = @kernel_equality;
    
    %m = x1'*x2;    
    %m = parse(x1,x2, kernel_polynomial(p));
    %m = parse(x1, x2, @kernel_hamming);
    %m = parse(x1, x2, @kernel_equality);
    %m = parse(x1,x2, kernel_gaussian(s));
    m = parse(x1,x2, kernel_exponential(s));    
    %m = parse(x1,x2, kernel_tanimoto_jaccard_coefficient());
    %m = parse(x1,x2, @(x1, x2) k1(x1,x2) + k2(x1,x2));
end

function k = kernel_polynomial(p)
    
    assert( p > 0, 'What are you doing!?');
    
    if p == 1
        k = @(x1,x2) x1'*x2;
    else
        k = @(x1,x2) 2^(-p)*(kernel_hamming(x1,x2)/numel(x1) + 1)^p;
    end
end

function k = kernel_sigmoid()
    k = @(x1,x2) tanh(x1'*x2);
end

function k = kernel_exponential(s)
    k = @(x1,x2) exp(-norm(x1-x2)/s);
end

function k = kernel_gaussian(s)
    k = @(x1,x2) exp(-sum_square(x1-x2)/s);
end

function k = kernel_hamming(x1, x2)   
    k = x1'*x2 + (x1-1)'*(x2-1);
end

function k = kernel_equality(x1, x2)
    k = all(x1 == x2);
end

function k = kernel_tanimoto_jaccard_coefficient()
    k = @(x1,x2) (x1'*x2)/(x1'*x2 + sum(abs(x1-x2)));
end
%Kernels
