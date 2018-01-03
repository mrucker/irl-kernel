% Abbeel & Ng algorithm implementation (projection version).
function irl_result = algorithm5run(algorithm_params,mdp_data,mdp_model,feature_data,example_samples,true_features,verbosity)

    % Fill in default parameters.
    algorithm_params = algorithm5defaultparams(algorithm_params);

    % Set random seed.
    if algorithm_params.seed ~= 0
        rng(algorithm_params.seed);
    end

    tic;

    % Initialize variables.
    [states,actions,~] = size(mdp_data.sa_p);
    [N,T] = size(example_samples);

    % Build feature membership matrix.
    if algorithm_params.all_features
        F = feature_data.splittable;
        %F = horzcat(F,ones(states,1));
    elseif algorithm_params.true_features
        F = true_features;
    else
        F = eye(states);
    end;

    % Count features
    F = F';
    features = size(F,1);

    % Construct state expectations
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
    mE = F*sE;

    %draw(sE, nE);
    
    % Generate random policy.
    rand_r = rand(states,1);
    rand_p = standardmdpsolve(mdp_data, rand_r);
    rand_s = standardmdpfrequency(mdp_data, rand_p);
    rand_m = F*rand_s;
    
    % Initialize t.
    rs = {rand_r};
    ps = {rand_p};
    ss = {rand_s};
    ms = {rand_m};
    sb = {rand_s};
    mb = {rand_m};
    
    ts = {0};
    tm = {0};

    ff = k(F,F, algorithm_params);
    ff = ff./ff(1,1);

    i = 2;

    rs{i} = ff*(sE-sb{i-1});
    ps{i} = standardmdpsolve(mdp_data,repmat(rs{i},1,actions));
    ss{i} = standardmdpfrequency(mdp_data, ps{i});
    ms{i} = F*ss{i};
    
    tm{i} = norm(mE - mb{i-1});
    ts{i} = sqrt(sE'*ff*sE + sb{i-1}'*ff*sb{i-1} - 2*sE'*ff*sb{i-1});
    
    fprintf(1,'Completed IRL iteration, i=%d, t=%f\n',i,ts{i});
    
    i = 3;

    while 1

        % Compute t and w using projection.
        sn     = (ss{i-1}-sb{i-2})'*ff*(sE-sb{i-2});
        sd     = (ss{i-1}-sb{i-2})'*ff*(ss{i-1}-sb{i-2});
        sc      = sn/sd;
        sb{i-1} = sb{i-2} + sc*(ss{i-1}-sb{i-2});
        
        mn     = (ms{i-1}-mb{i-2})'*(mE-mb{i-2});
        md     = (ms{i-1}-mb{i-2})'*(ms{i-1}-mb{i-2});
        mc      = mn/md;
        mb{i-1} = mb{i-2} + mc*(ms{i-1}-mb{i-2});

        % Recompute optimal policy using new weights.
        rs{i} = ff*(sE-sb{i-1});
        ps{i} = standardmdpsolve(mdp_data,repmat(rs{i},1,actions));
        ss{i} = standardmdpfrequency(mdp_data, ps{i});
        ms{i} = F*ss{i};
        
        ts{i} = sqrt(sE'*ff*sE + sb{i-1}'*ff*sb{i-1} - 2*sE'*ff*sb{i-1});
        tm{i} = norm(mE - mb{i-1});

        if verbosity ~= 0
            fprintf(1,'Completed IRL iteration, i=%d, t=%f\n',i,ts{i});
        end;
        
        if (abs(ts{i}-ts{i-1}) <= algorithm_params.epsilon)
            break;
        end;

        i = i + 1;

    end;

    idx = i;
    
    % Solve optimization to determine lambda weights.
    % In Abbeel & Ng's algorithm, we should use the weights lambda to construct
    % a stochastic policy. However, here we are evaluating IRL algorithms, so
    % we must return a single reward. To this end, we'll simply pick the reward
    % with the largest weight lambda.
    [~,idx] = max(mixPolicies_1(sE, ss, rs, ff));
    %[~,idx] = max(mixPolicies_2(mE, ms, ff, verbosity));

    time = toc;
    
    t  = ts{idx};
    r  = rs{idx};
    
    if verbosity ~= 0
        fprintf(1,'FINISHED IRL,i=%d, t=%f \n',idx,t);
    end
    
    irl_result = marshallResults(repmat(r, 1, actions), 0, mdp_model, mdp_data, time);
end

function [lambda] = mixPolicies_1(sE, ss, rs, ff)
    s_mat = cell2mat(ss);
    r_mat = cell2mat(rs);
    
    f_cnt = size(s_mat,1);
    s_cnt = size(s_mat,2);
        
    ssffss = s_mat'*ff*s_mat;
    seffse = sE'*ff*sE;
    seffss = sE'*ff*s_mat;
     
    %sd = diag(s_mat'*s_mat + sE'*sE - 2*s_mat'*sE); Didn't seem to work well
    %fd = diag(ssffss + seffse - 2*seffss); Didn't seem to work well
    %rd = diag(r_mat'*(sE - s_mat)); Didn't seem to work well
    
    % Solve optimization to determine lambda weights.
    cvx_begin
        cvx_quiet(true);
        variables l(s_cnt);
        minimize(l'*ssffss*l + seffse - 2*seffss*l);
        subject to
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

function k = k(x1, x2, params)
    p = params.p;
    s = params.s;
    c = params.c;
    n = size(x1,1);
        
    switch params.k
        case 1
            b = k_dot();
        case 2
            b = k_polynomial(k_hamming(1),p,c);
        case 3
            b = k_hamming();
        case 4
            b = k_equal(k_norm());
        case 5
            b = k_gaussian(k_hamming(0),s);
        case 6
            b = k_exponential(k_hamming(0),s);
        case 7
            b = k_anova(n);
        case 8
            b = k_exponential_compact(k_norm(),s);
    end
       
    k = b(x1,x2);
end