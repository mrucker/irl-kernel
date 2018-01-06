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
    [states,actions,~] = size(mdp_data.sa_p);

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

    %true_r = algorithm_params.true_r;
    %true_p = standardmdpsolve(mdp_data, true_r);
    %sE2    = standardmdpfrequency(mdp_data, true_p);    

    %draw(sE, sE2, 'Discounted Visits', 'True Frequency');
    
    % Step 1
    rand_r = rand(states,1);
    rand_p = standardmdpsolve(mdp_data, repmat(rand_r, 1, actions));
    rand_s = standardmdpfrequency(mdp_data, rand_p);

    rs = {rand_r};
    ps = {rand_p};
    ss = {rand_s};    
    ts = {0};    

    i = 2;
    
    ff = k(F,F, algorithm_params);    
    
    while 1

        x = horzcat(sE, cell2mat(ss));
        y = vertcat(1,-ones(i-1,1));

        %[t1, g1, b1, u1, r1] = maxMarginOptimization_1_h(y, x, ff, verbosity);
        [t, g, b, u, r]      = maxMarginOptimization_4_s(y, x, ff, verbosity);

        rs{i} = r;
        ps{i} = standardmdpsolve(mdp_data, repmat(rs{i}, 1, actions));
        ss{i} = standardmdpfrequency(mdp_data, ps{i});
        ts{i} = t;        
        
        rd = r'*(sE - ss{i});
        
        if verbosity ~= 0
            fprintf(1,'Completed IRL iteration, i=%d, t=%f\n',i,t);
        end
        
        %Running for too long after converging causes the selection of features weights at the end to be very problematic.
        %It is much better, in general, to exit quickly after convergence than to keep toiling away.
        %In other words, the final result is very sensitive to exit conditions.
        if (ts{i} <= algorithm_params.epsilon || abs(ts{i}-ts{i-1}) <= algorithm_params.epsilon || (i > 2 && ts{i}-ts{i-1} > ts{i-1}*2))
            break;
        end

        i = i+1;
    end
    
    idx = i;
    
    % In Abbeel & Ng's algorithm, we should use the weights lambda to construct
    % a stochastic policy. However, here we are evaluating IRL algorithms, so
    % we must return a single reward. To this end, we'll simply pick the reward
    % with the largest weight lambda.
    ls = mixPolicies_1(sE, ss, rs, ff);
    [~,idx] = max(ls);
    %[~,idx] = max(mixPolicies_2(mE, ms, ff, verbosity));

    time = toc;
    
    t = ts{idx};
    r = rs{idx};
    
    if verbosity ~= 0
        fprintf(1,'FINISHED IRL, i=%d, t=%f\n',idx,t);
    end
    
    %draw_visitation(sE, ss{idx}, 'sE', 'ss{idx}');
    
    irl_result = marshallResults(repmat(r, 1, actions), 0, mdp_model, mdp_data, time);
end

function [margin, right, wrong, unknown, reward] = maxMarginOptimization_4_s(y, x, ff, varargin)
    o_cnt = size(x,2);
    s_cnt = size(x,1);    

    vv = x'*ff*x;

    warning('off','all')
    cvx_begin
        cvx_quiet(true);
        variables a(o_cnt);
        maximize(sum(a) - 1/2*quad_form(a.*y, vv)) %dual problem
        subject to
        %I don't understand it, but this constraint seems to make things worse. I ran about 20 tests with and without it to confirm.
        %It doesn't always make it worse, but when it does it results in a catastrophic failure. 
        %Scratch that. With enough data it makes things better. It seems to somehow be tied into the bias        
            0 == a'*y;
            0 <= a;
    cvx_end
    warning('off','all')
    
    %sv = x(:,round(a,8)>0);
    %sl = y(round(a,8)>0,1);
    
    %regarding b0: "we typically use an average of all the solutions for numerical stability" (ESL pg.421)
    %b0 = mean(sl - sv'*ff*x*(a.*y)); %aka , -(a'*vv*(a.*y)/sum(a)); Doesn't seem to make a difference to performance
    %rs = ff*x*(a.*y) + b0; including b0 in reward calculation doesn't seem to make a demonstrable difference
    rs = ff*x*(a.*y);
    
    %ds      = sign(y.*(vv*(a.*y) + b0));
    ds      = zeros(size(x,2),1);
    right   = sum(ds == 1);
    wrong   = sum(ds == -1);
    unknown = sum(ds == 0);
    reward  = rs;
    margin  = (1/sqrt(sum(a)));
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

%Drawing
function draw_visitation(r1, r2, t1, t2)

n = sqrt(size(r1,1));

% Create final figure.
figure('Name','visitation comparison','NumberTitle','off');
hold on;
grid on;
cla;

% Draw reward for ground truth.
subplot(1,2,1);
title(t1);
feval(strcat('gridworld','draw'), r1, [], [], struct('n',n));

% Draw reward for IRL result.
subplot(1,2,2);
title(t2);
feval(strcat('gridworld','draw'), r2, [], [], struct('n',n));

% Turn hold off.
hold off;
end
%Drawing

%Kernels
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
            b = k_exponential(k_norm(),s);
        case 7
            b = k_anova(n);
    end
       
    k = b(x1,x2);
end
%Kernels
