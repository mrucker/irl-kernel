add_ourpaths; %includes all the paths required to run our code

%close all; clear; clc;

algorithm       = 'gpirl';
%algorithm       = 'maxent';
%algorithm       = 'mmpboost';
%algorithm       = 'learch';
algorithm       = 'an';
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
algorithm_params = struct('all_features',0 , 'true_features',1 , 'epsilon', .0001, 'k',5, 's',.0001, 'p',30, 'c',.9);
mdp_model        = 'standardmdp';%'linearmdp' (stochastic) or 'standardmdp' (deterministic)
mdp              = 'peterworld';
mdp_params       = struct('n',10, 'determinism',1, 'seed',seed, 'b',2, 'discount',.9);
test_params      = struct('training_sample_lengths', 100, 'training_samples', 800, 'verbosity',1);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

fprintf(1,'Seed:%10f\n', mdp_params.seed);

%Visualize solution.
printresult(test_result);
%visualize(test_result, 1, strcat(algorithm, '-- k', num2str(algorithm_params.k)), algorithm_params);

%Comparison%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%algorithm_params.k = 6;
%test_result = runtest('algorithm3', algorithm_params, mdp_model, mdp, mdp_params, test_params);
%visualize(test_result, 1, 'algorithm3 -- k6', algorithm_params);

