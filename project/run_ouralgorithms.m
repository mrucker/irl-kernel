add_ourpaths; %includes all the paths required to run our code

%close all; clear; clc;

%algorithm       = 'gpirl';
%algorithm       = 'maxent';
%algorithm       = 'mmpboost';
%algorithm       = 'learch';
algorithm       = 'an';
%algorithm       = 'algorithm3';
algorithm       = 'algorithm5';
%algorithm       = 'mmp';
%algorithm       = 'firl';

seed = sum(100*clock);
%seed = 208738.400000;
%seed = 205137.000000;
seed = 208050.800000;

%all_features = bad features. true_features = perfect features. (0,0) = good features.
algorithm_params = struct('all_features',1 , 'true_features',0 , 'epsilon', .000001, 'k',6, 's',.1, 'p', 1, 'c', 1);
mdp_model        = 'standardmdp';%'linearmdp' (stochastic) or 'standardmdp' (deterministic)
mdp              = 'gridworld';%sum(100*clock)%208738.400000
mdp_params       = struct('n',10, 'determinism',1, 'seed',seed, 'b',2, 'discount',.9);
test_params      = struct('training_sample_lengths', 2^7, 'training_samples', 2^10, 'verbosity',1);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

fprintf(1,'Seed:%10f\n', mdp_params.seed);

%Visualize solution.
printresult(test_result);
visualize(test_result, 1, strcat(algorithm, '-- k', num2str(algorithm_params.k)), algorithm_params);

%Comparison%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%algorithm_params.k = 6;
%test_result = runtest('algorithm3', algorithm_params, mdp_model, mdp, mdp_params, test_params);
%visualize(test_result, 1, 'algorithm3 -- k6', algorithm_params);

