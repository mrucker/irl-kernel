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
    sE   = zeros(states,1);
    
    for i=1:N
        for t=1:T
           sE(example_samples{i,t}(1)) = sE(example_samples{i,t}(1)) + 1*mdp_data.discount^(t-1);
        end
    end

    sE = sE/N;

    true_r = algorithm_params.true_r;
    true_p = standardmdpsolve(mdp_data, true_r);
    %sE     = standardmdpfrequency(mdp_data, true_p);
    mE     = F*sE;

    % Step 1
    rand_w = rand(features,1);
    rand_w = rand_w/norm(rand_w); %not important to the algorithm, but this allows for comparisons against future w's.
    rand_r = repmat(F'*rand_w, 1, actions);
    rand_p = standardmdpsolve(mdp_data, rand_r);
    rand_s = standardmdpfrequency(mdp_data, rand_p);
    rand_m = F*rand_s;

    rs = {rand_r};
    ps = {rand_p};
    ss = {rand_s};
    ms = {rand_m};
    ts = {0};    

    i = 2;
    
    while 1

        x = horzcat(sE, cell2mat(ss));        
        y = vertcat(1,-ones(i-1,1));
        
        %Step 2
        [m, r, w, ~, d] = maxMarginOptimization_4_s(y, x, F, verbosity, algorithm_params.p, algorithm_params.s);
        
        % Print t.
        if verbosity ~= 0
            fprintf(1,'Completed IRL iteration, t=%f, r=%d, w=%d\n',m,r,w);
        end
                
        rs{i} = repmat(d(F), 1, actions);
        ts{i} = m;
        
        %Step 3
        if (i==40 || ts{i} <= algorithm_params.epsilon || abs(ts{i}-ts{i-1}) <= algorithm_params.epsilon || (i > 2 && ts{i}-ts{i-1} > ts{i-1}*2))
            break;
        end

        %Step 4        
        ps{i} = standardmdpsolve(mdp_data, rs{i});
        ss{i} = standardmdpfrequency(mdp_data, ps{i});        
        ms{i}  = F*ss{i};

        %Step 6
        i = i+1;
    end
    
    % Compute mu for last policy    
    ps{i} = standardmdpsolve(mdp_data, rs{i});
    ms{i} = F*standardmdpfrequency(mdp_data, ps{i});
    fs{i} = standardmdpfrequency(mdp_data, ps{i});

    % In Abbeel & Ng's algorithm, we should use the weights lambda to construct
    % a stochastic policy. However, here we are evaluating IRL algorithms, so
    % we must return a single reward. To this end, we'll simply pick the reward
    % with the largest weight lambda.
    [~,idx] = max(mixPolicies(mE, ms, verbosity));

    time = toc;
    
    %idx = i;
    
    %d = cell2mat(ts);
    %idx = find(d == min(d(2:end)));

    %irl_result = marshallResults(rs{idx}, ws{idx}, mdp_model, mdp_data, time);
    irl_result = marshallResults(rs{idx}, 0, mdp_model, mdp_data, time);
end

function [margin, right, wrong, unknown, dk] = maxMarginOptimization_4_s(y, x, F, verbosity, varargin)
    f_cnt = size(x,1);
    o_cnt = size(x,2);    
    
    
    z = x'*k(F,F, varargin)*x;
    
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables a(o_cnt);
        maximize(sum(a) - 1/2*quad_form(a.*y, z)) %dual problem
        subject to
            0 == a'*y;
            0 <= a;
    cvx_end
    warning('off','all')
    
    %Useful to study kernel polynomials of degree 2. This is the new feature space we are working in.
    %f_x = poly_f(x,1);
    
    %When working in higher dimensions I'm not sure this has any meaning. (such as polynomial kernels above 1).
    %When working within the dimensions of x this is the normal to the hyperplane.
    %f_w = f_x*(a.*y);
    %f_m = 1/norm(f_w);
    
    %regarding b0: "we typically use an average of all the solutions for numerical stability" (ESL pg.421)
    %b0 = sum(y - k(z, z)*(a.*y))/sum(a>0);
    b0 = (sum(y.*(a>0)) - sum(z*(a.*y)))/sum(a>0);
    
    dk = @(m) k(m,F, varargin)*x*(a.*y) + b0;

    margin  = 1/sqrt(sum(a));
    
    %ds = sign(y.*(z*(a.*y) + b0));
    ds = zeros(size(x,2),1);
    right   = sum(ds == 1);
    wrong   = sum(ds == -1);
    unknown = sum(ds == 0);
end

function [lambda] = mixPolicies(mE, ms, verbosity)

    f_cnt = size(ms{1},1);
    m_cnt = size(ms,2);

    % Construct matrix.
    m_mat = zeros(f_cnt,m_cnt);
    for j=1:m_cnt
        m_mat(:,j) = ms{j};
    end

    % Solve optimization to determine lambda weights.
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variable m(f_cnt);
        variable l(m_cnt);
        minimize(norm(m-mE));
        subject to            
            m == m_mat*l;
            l >= zeros(m_cnt,1);
            1 >= norm(l,1);
            1 == ones(m_cnt,1)'*l;
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

%Kernels
function m = parse(x1, x2, k)
    m = zeros(size(x1,1),size(x2,2));
    
    x1 = x1';
    
    for c = 1:size(x2,2)
        for r = 1:size(x1,1)
            m(r,c) = k(x1(r,:)', x2(:,c));
        end
    end
end

function m = k(x1, x2, params)
    p = params{1};
    s = params{2};
    
    %m = kernel_poly(x1,x2,p);
    %m = parse(x1,x2, kernel_gaussian(s));
    m = parse(x1,x2, kernel_exponential(s));    
    %m = parse(x1,x2, kernel_tanimoto_jaccard_coefficient());
end

function k = kernel_poly(x1, x2, p)
    
    assert( p > 0, 'What are you doing!?');

    if p == 1
        k = x1'*x2;
    else
        k = power(x1'*x2 + ones(size(x1,2), size(x2,2)), p*ones(size(x1,2), size(x2,2)));
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

function k = kernel_tanimoto_jaccard_coefficient()
    k = @(x1,x2) (x1'*x2)/(x1'*x2 + sum(abs(x1-x2)));
end
%Kernels
