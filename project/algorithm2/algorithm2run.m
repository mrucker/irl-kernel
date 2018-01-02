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
    
    F = F';
    F_0 = F; 
    F = poly_f(F,2)./2; %convert to the feature space of our kernel

    ff = k(F_0,F_0, algorithm_params);
    ff = ff./ff(1,1);
    
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
    rand_w = rand(size(F_0,1),1);
    rand_w = rand_w/norm(rand_w); %not important to the algorithm, but this allows for comparisons against future w's.
    rand_r = F_0'*rand_w;
    rand_p = standardmdpsolve(mdp_data, repmat(rand_r, 1, actions));
    rand_s = standardmdpfrequency(mdp_data, rand_p);
    rand_m = F*rand_s;

    rs = {rand_r};
    ps = {rand_p};
    ss = {rand_s};
    ms = {rand_m};
    ts = {0};    

    i = 2;
    
    while 1

        m_cnt = i-1;
        f_cnt = size(mE,1);

        mx = horzcat(mE, cell2mat(ms));
        sx = horzcat(sE, cell2mat(ss));
        y = vertcat(1,-ones(m_cnt,1));
        
        
        %Step 2
        [m0,g0,b0,u0,r0] = maxMarginOptimization_1_h(y, mx, F, verbosity);
        [m1,g1,b1,u1,r1] = maxMarginOptimization_4_s(y, sx, ff, mx, F, verbosity);
        
        rs{i} = r1;
        ts{i} = m1;

        if verbosity ~= 0
            fprintf(1,'Completed IRL iteration, t=%f\n',ts{i});
        end
        
        %Step 3
        if (ts{i} <= algorithm_params.epsilon || abs(ts{i}-ts{i-1}) <= algorithm_params.epsilon || (i > 2 && ts{i}-ts{i-1} > ts{i-1}*2))
            break;
        end

        %Step 4        
        ps{i} = standardmdpsolve(mdp_data, repmat(rs{i}, 1, actions));
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
    ls = mixPolicies(mE, ms, verbosity);
    [~,idx] = max(ls);

    time = toc;
    
    %idx = i;
    
    %d = cell2mat(ts);
    %idx = find(d == min(d(2:end)));

    %irl_result = marshallResults(rs{idx}, ws{idx}, mdp_model, mdp_data, time);
    irl_result = marshallResults(rs{idx}, 0, mdp_model, mdp_data, time);
end

function [margin, right, wrong, unknown, reward] = maxMarginOptimization_1_h(y, mx, f, verbosity, varargin)
    f_cnt = size(mx,1);
    o_cnt = size(mx,2);
        
    warning('off','all')
    cvx_begin
        cvx_quiet(true);
        variables m w(f_cnt);
        maximize(m);
        subject to
            1 >= norm(w);
            m <= y.*(mx'*w);
    cvx_end
    warning('off','all')
        
    %ds = y.*(x'*w);
    ds = zeros(size(mx,2),1);
    
    margin  = m;
    right   = sum(sign(ds) == 1);
    wrong   = sum(sign(ds) == -1);
    unknown = sum(sign(ds) == 0);
    reward  = f'*w;
end

function [margin, right, wrong, unknown, reward] = maxMarginOptimization_4_s(y, sx, ff, mx, f, verbosity, varargin)
    o_cnt = size(sx,2);
    s_cnt = size(sx,1);
    %f_cnt = size(mx,1);

    vv = sx'*ff*sx;
    
    warning('off','all')
    cvx_begin
        cvx_quiet(true);
        variables a1(o_cnt);
        maximize(sum(a1) - 1/2*quad_form(a1.*y, vv)) %dual problem
        subject to
            %0 == a1'*y;
            0 <= a1;
    cvx_end
    warning('off','all')

%     cvx_begin
%         cvx_quiet(true);
%         variables a2(o_cnt);
%         maximize(sum(a2) - 1/2*quad_form(a2.*y, vv)) %dual problem
%         subject to
%             0 == a2'*y;
%             0 <= a2;
%     cvx_end
%     warning('off','all')
%         
%     warning('off','all')
%     cvx_begin
%         cvx_quiet(true);
%         variables m w(f_cnt);
%         maximize(m);
%         subject to
%             1 >= norm(w);
%             m <= y.*(mx'*w);
%     cvx_end
%     warning('off','all')
    
    sv = sx(:,round(a1,8)>0);
    sl = y(round(a1,8)>0,1);
    
    %regarding b0: "we typically use an average of all the solutions for numerical stability" (ESL pg.421)
    b0 = mean(sl - sv'*ff*sx*(a1.*y)); %aka , -(a'*vv*(a.*y)/sum(a));
    rs = ff*sx*(a1.*y)+b0;
    
    %ds      = sign(y.*(vv*(a.*y) + b0));
    ds      = zeros(size(sx,2),1);
    right   = sum(ds == 1);
    wrong   = sum(ds == -1);
    unknown = sum(ds == 0);
    reward  = rs;
    margin  = 1/sqrt(sum(a1));
end

function [lambda] = mixPolicies(mE, ms, verbosity)

    f_cnt = size(mE,1);
    m_cnt = size(ms,2);    
    m_mat = cell2mat(ms);

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
            l >= 0;
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

function features = poly_f(F, p)

    if (p == 1)
        features = F;
    else
        features = [];

        for i = 1:size(F,2)
            n1 = [1];
            n2 = [sqrt(2)*(F(:,i))];
            n3 = [(F(:,i)).*(F(:,i))];

            n4=[];
            for j = 1:size(F,1)
                for k = (j+1):size(F,1)
                    n4 = vertcat(n4,[sqrt(2)*F(j,i)*F(k,i)]);
                end
            end
            features = horzcat(features, vertcat(n1, n2, n3, n4));
        end
    end
end

function k = k(x1, x2, params)
    p = params.p;
    s = params.s;
    c = 1;
    n = size(x1,1);
        
    %b = k_dot();
    b = k_polynomial(k_dot(),p,c);
    %b = k_hamming();
    %b = k_equal(k_norm());
    %b = k_gaussian(k_norm(),s);
    %b = k_exponential(k_norm(),s);
    %b = k_anova(n);
    
    k = b(x1,x2);
end
