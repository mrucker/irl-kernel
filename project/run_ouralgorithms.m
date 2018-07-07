add_ourpaths; %includes all the paths required to run our code

%close all; clear; clc;

%algorithm       = 'gpirl';
%algorithm       = 'maxent';
%algorithm       = 'mmpboost';
%algorithm       = 'learch';
%algorithm       = 'an';
%algorithm       = 'algorithm3'; %Support Vector 'an' algorithm with kernel
algorithm       = 'algorithm5'; %Projection     'an' algorithm with kernel #1
%algorithm       = 'algorithm6'; %Value-based    'an' algorithm with kernel #2
%algorithm       = 'algorithm7'; %algorithm6 with (Klein 2012) flavor.      #4
%algorithm       = 'algorithm8'; %algorithm6 with a single step update.     #3
%algorithm       = 'mmp';
%algorithm       = 'firl';

seed = sum(100*clock);
%seed = 208738.400000;
%seed = 205137.000000;
%seed = 208050.800000;

%all_features = bad features. true_features = perfect features. (0,0) = good features.
algorithm_params = struct('all_features',0 , 'true_features',1 , 'epsilon', .01, 'k',1, 's',.0001, 'p',30, 'c',.9);
mdp_model        = 'standardmdp';%'linearmdp' (stochastic) or 'standardmdp' (deterministic)
mdp              = 'gridworld';

mdp_params       = struct('n',200, 'determinism',1, 'seed',seed, 'b',2, 'discount',.9);
test_params      = struct('training_sample_lengths', 200, 'training_samples', 800, 'verbosity',1, 'true_examples', {[]});

%take this out if not running against peterworld
%mdp              = 'peterworld';
%test_params.true_examples = {[10, 11, 3, 9, 12, 3, 3, 9, 7, 5, 5, 3, 4, 10, 10, 10, 10, 5, 10, 3, 3, 4, 10, 10, 1, 7, 10, 11, 7, 1, 5, 4, 10, 10, 1, 7, 10, 5, 10, 7, 10, 5, 1, 7, 10, 10, 10, 10, 1, 8, 1, 4, 10, 10, 10, 10, 5, 4, 5, 4, 10, 10, 5, 4, 10, 6, 4, 10, 10, 10, 6, 4, 10, 10, 6, 4, 10, 10, 10, 10, 1, 4, 5, 10, 10, 10, 1, 4, 10, 10, 10, 5, 4, 10, 1, 5, 1, 4, 10, 10, 10, 10, 5, 5, 4, 10, 10, 5, 4, 10, 10, 1, 4, 10, 10, 1, 2, 4, 10, 10, 5, 4, 1, 4, 2, 1, 5, 4, 10, 10, 10, 5, 4, 10, 10, 1, 1, 1, 8, 8, 4, 10, 10, 6, 4, 10, 10, 5, 4, 10, 1, 5, 4, 10, 10, 10, 10, 10, 10, 5, 4, 10, 10, 10, 1, 2, 4, 5, 10, 5, 4, 10, 10, 5, 4, 10, 10, 10, 10, 10, 1, 4, 1, 1, 4, 10, 11, 7, 11, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 4, 5, 4, 10, 10, 5, 4, 10, 1, 5, 6, 4, 4, 10, 10, 10, 10, 1, 4, 10, 10, 10, 11, 7, 10, 10, 1, 5, 10, 10, 10, 10, 10, 10, 10, 10, 1, 5, 10, 10, 1, 1, 8, 4, 4, 10, 10, 6, 4, 10, 10, 10, 10, 1, 4, 10, 10, 1, 4, 10, 10, 10, 10, 10, 10, 10, 1, 4, 10, 10, 10, 10, 10, 6, 4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6, 1, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 4, 10, 6, 4, 10, 6, 4, 10, 1, 5, 4, 10, 10, 5, 4, 10, 10, 10, 10, 10, 6, 4, 1, 4, 11, 7, 10, 10, 5, 4, 6, 4, 10, 10, 10, 10, 1, 7, 10, 10, 6, 4, 10, 6, 3, 3, 4, 10, 10, 5, 4, 10, 10, 1, 7, 1, 4, 1, 7, 10, 10, 10, 1, 5, 10, 10, 6, 9, 1, 4, 7, 4, 10, 1, 5, 1, 4, 10, 10, 6, 4, 6, 4, 6, 1, 4, 10, 10, 10, 10, 10, 1, 4, 10, 10, 11, 7, 11, 7, 10, 10, 10, 6, 4, 10, 10, 1, 4, 5, 10, 10, 10, 5, 4, 10, 1, 1, 4, 10, 10, 1, 5, 10, 1, 5, 10, 10, 10, 10, 10, 1, 4, 10, 10, 10, 6, 4, 10, 10, 6, 4, 10, 10, 10, 10, 1, 4, 10, 5, 1, 1, 2, 4, 10, 10, 1, 7, 6, 4, 10, 10, 10, 10, 1, 7, 10, 10, 10, 10, 5, 4, 10, 11, 7, 10, 6, 3]};

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

fprintf(1,'Seed:%10f\n', mdp_params.seed);

%Visualize solution.
printresult(test_result);
%visualize(test_result, 1, strcat(algorithm, '-- k', num2str(algorithm_params.k)), algorithm_params);

%Comparison%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%algorithm_params.k = 6;
%test_result = runtest('algorithm3', algorithm_params, mdp_model, mdp, mdp_params, test_params);
%visualize(test_result, 1, 'algorithm3 -- k6', algorithm_params);

