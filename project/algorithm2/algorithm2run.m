function irl_result = algorithm2run(algorithm_params, mdp_data, mdp_model, feature_data, example_samples, true_features, verbosity)

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
    algorithm_params = algorithm2defaultparams(algorithm_params);

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
    features = size(F,2);
    F = F';


    % Construct state expectations.
    mE   = zeros(features,1);    
    
    for i=1:N
        for t=1:T
            mE = mE + F(:, example_samples{i,t}(1)) * mdp_data.discount^(t-1);
        end
    end
    
    mE = mE/N;

    % Step 1
    rand_w = rand(features,1);
    rand_w = rand_w/norm(rand_w); %not important to the algorithm, but this allows for comparisons against future w's.
    rand_r = repmat(F'*rand_w, 1, actions);
    rand_p = standardmdpsolve(mdp_data, rand_r);
    rand_m = F*standardmdpfrequency(mdp_data, rand_p);

    rs = {rand_r};
    ps = {rand_p};
    ms = {rand_m};
    ts = {0};    

    i = 2;

    while 1        
        
        %Step 2
        %[r1, t1] = maxMarginOptimization_1_a(mE, ms, F, actions, verbosity);
        %[r2, t2] = maxMarginOptimization_2_a(mE, ms, F, actions, verbosity);
        %[r3, t3] = maxMarginOptimization_3_a(mE, ms, F, actions, verbosity);
        [rs{i}, ts{i}] = maxMarginOptimization_1_a(mE, ms, F, actions, verbosity, algorithm_params.p);
        
        % Print t.
        if verbosity ~= 0
         %   fprintf(1,'Completed IRL iteration, w=(%f,%f), t=(%f,%f)\n',w1(1)/ws{i}(1),w1(2)/ws{i}(2),ts{i},t1(1));
            fprintf(1,'Completed IRL iteration, t=%f\n',ts{i});
        end
        
        %Step 3
        if (ts{i} <= algorithm_params.epsilon || abs(ts{i}-ts{i-1}) <= algorithm_params.epsilon || (i >= 5 && abs(ts{i}-ts{i-1}) > 100000))
            break;
        end

        %Step 4        
        ps{i} = standardmdpsolve(mdp_data, rs{i});

        %Step 5
        ms{i} = F*standardmdpfrequency(mdp_data, ps{i});

        %Step 6
        i = i+1;
    end

    % Compute mu for last policy    
    ps{i} = standardmdpsolve(mdp_data, rs{i});
    ms{i} = F*standardmdpfrequency(mdp_data, ps{i});
    
    % In Abbeel & Ng's algorithm, we should use the weights lambda to construct
    % a stochastic policy. However, here we are evaluating IRL algorithms, so
    % we must return a single reward. To this end, we'll simply pick the reward
    % with the largest weight lambda.
    [~,idx] = max(mixPolicies(mE, ms, verbosity));

    time = toc;
    
    idx = i-2;

    %irl_result = marshallResults(rs{idx}, ws{idx}, mdp_model, mdp_data, time);
    irl_result = marshallResults(rs{idx}, 0, mdp_model, mdp_data, time);
end

%a version of the hard-margin SVM where the margin is directly optimized
function [r,t] = maxMarginOptimization_1_a(mE, ms, F, actions, verbosity, varargin)
    f_cnt = size(mE,1);
    m_cnt = size(ms,2);
    
    % Construct matrix.
    m_mat = zeros(f_cnt,m_cnt);
    for j=1:m_cnt
        m_mat(:,j) = ms{j};
    end
    
    k = @(x1,x2) x1'*x2;
    
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables t w(f_cnt);
        maximize(t);
        subject to
            1 >= k(w,w);
            t <= k(w,mE) - k(w,m_mat);
    cvx_end
    warning('off','all')
    
    r = repmat(F'*w, 1, actions);
end

%standard version soft-max which allowing some error in the decision boundary
function [r,t] = maxMarginOptimization_2_a(mE, ms, F, actions, verbosity, varargin)
    f_cnt = size(ms{1},1);
    m_cnt = size(ms,2);
    
    % Construct matrix.
    m_mat = horzcat(mE, zeros(f_cnt,m_cnt));
    for j=1:m_cnt
        m_mat(:,j+1) = ms{j};
    end
    
    y = vertcat(1,-ones(m_cnt,1));
    
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables w_0 w(f_cnt) zeta(m_cnt+1);
        minimize(norm(w,2) + 100*sum(zeta));
        subject to
            1 <= y.*(m_mat'*w + w_0) + zeta;
            0 <= zeta;
    cvx_end
    warning('off','all')
    
    r = repmat(F'*w, 1, actions);
    t = 2*(1/norm(w)); % we multiply by two to get both margins.
end

%another version of soft-max using the hinge-loss objective. 
function [r,t] = maxMarginOptimization_3_a(mE, ms, F, actions, verbosity, varargin)
    f_cnt = size(ms{1},1);
    m_cnt = size(ms,2);
    
    % Construct matrix.
    m_mat = horzcat(mE, zeros(f_cnt,m_cnt));
    for j=1:m_cnt
        m_mat(:,j+1) = ms{j};
    end
    
    y = vertcat(1,-ones(m_cnt,1));
    
    k = @(x1,x2) x1'*x2;
    
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables w_0 w(f_cnt);
        minimize( 100*sum(max(0,1-y.*(k(m_mat,w) + w_0))) + k(w,w))
    cvx_end
    warning('off','all')
    
    r = repmat(F'*w, 1, actions);
    t = 2*(1/norm(w)); % we multiply by two to get both margins.
end

%another version of soft-max using the hinge-loss objective. 
function [r,t] = maxMarginOptimization_5_a(mE, ms, F, actions, p, verbosity)
    f_cnt = size(ms{1},1);
    m_cnt = size(ms,2);
    
    % Construct matrix.
    m_mat = horzcat(mE, zeros(f_cnt,m_cnt));
    for j=1:m_cnt
        m_mat(:,j+1) = ms{j};
    end
    
    y = vertcat(1,-ones(m_cnt,1));
    
    k = @(x1,x2) x1'*x2;
    k1 = @(x1,x2) (x1'*x2) + ones(size(x1,2));
    kp = @(x1,x2,p) power(k1(x1,x2), p*ones(size(x1,2)));
    
    warning('off','all')
    cvx_begin
        if verbosity == 2
            cvx_quiet(false);
        else
            cvx_quiet(true);
        end
        variables a(m_cnt+1);
        maximize(sum(a) - 1/2*quad_form(a.*y, kp(m_mat,m_mat,p)))
        subject to
            0 == a'*y;
            0 <= a;
    cvx_end
    warning('off','all')
        
    w = m_mat*(a.*y);
    r = repmat(F'*w, 1, actions);
    t = 2/norm(w); % we multiply by two to get both margins.
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
        minimize(sum_square(m-mE));
        subject to
            m == m_mat*l;
            l >= zeros(m_cnt,1);
            ones(m_cnt,1)'*l == 1;
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
