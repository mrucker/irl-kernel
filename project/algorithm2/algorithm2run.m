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
        % Note that we add a row of 1s to the feature matrix to ensure that we
        % can control the reward at every state.
        F = horzcat(F,ones(states,1));
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
    
    mE = mE/(N*T); %not exactly what the paper does but I think it works and matches our code library. I think the paper would just divide by N.

    % Step 1
    rand_w = rand(features,1);
    rand_r = repmat(F'*rand_w, 1, actions);
    rand_p = standardmdpsolve(mdp_data, rand_r);
    rand_m = F*standardmdpfrequency(mdp_data, rand_p);

    rs = {rand_r};
    ps = {rand_p};
    ws = {rand_w};
    ms = {rand_m};

    i = 2;

    while 1
        %Step 2
        [ws{i}, t] = maxMarginOptimization(mE, ms);

        %Step 3
        if (abs(t) <= algorithm_params.epsilon)
            break;
        end

        %Step 4
        rs{i} = repmat(F'*ws{i}, 1, actions);
        ps{i} = standardmdpsolve(mdp_data, rs{i});

        %Step 5
        ms{i} = F*standardmdpfrequency(mdp_data, ps{i});

        %Step 6
        i = i+1;

        % Print t.
        if verbosity ~= 0
            fprintf(1,'Completed IRL iteration, t=%f\n',t);
        end
    end

    % Compute mu for last policy    
    rs{i} = repmat(F'*ws{i}, 1, actions);
    ps{i} = standardmdpsolve(mdp_data, rs{i});
    ms{i} = F*standardmdpfrequency(mdp_data, ps{i});
    
    % In Abbeel & Ng's algorithm, we should use the weights lambda to construct
    % a stochastic policy. However, here we are evaluating IRL algorithms, so
    % we must return a single reward. To this end, we'll simply pick the reward
    % with the largest weight lambda.
    [~,idx] = max(policyMixing(mE, ms, verbosity));

    time = toc;

    irl_result = marshallResults(rs{idx}, ws{idx}, mdp_model, mdp_data, time);
end

function [w,t] = maxMarginOptimization(mE, ms)
    w = mE;
    t = 0;
end

function [lambda] = policyMixing(mE, ms, verbosity)

    f_cnt = size(ms{1},1);
    m_cnt = size(ms,2);

    % Construct matrix.
    m_mat = zeros(f_cnt,m_cnt);
    for j=1:m_cnt
        m_mat(:,j) = ms{j};
    end

    % Solve optimization to determine lambda weights.
    cvx_begin
        if verbosity ~= 0
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
    
    lambda = lm;
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
