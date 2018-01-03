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

%all_features = bad features. true_features = perfect features. (0,0) = good features.
algorithm_params = struct('all_features',1 , 'true_features',0 , 'epsilon', .00001, 'k',6, 'p', 2, 's',.000001, 'c', 1);
mdp_model        = 'linearmdp';%'linearmdp' (stochastic) or 'standardmdp' (deterministic)
mdp              = 'gridworld';%sum(100*clock)
mdp_params       = struct('n',10, 'determinism',1, 'seed',208738.400000, 'b',2, 'discount',.9);
test_params      = struct('training_sample_lengths', 128, 'training_samples', 2^10, 'verbosity',1);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

fprintf(1,'Seed:%10f\n', mdp_params.seed);

%Visualize solution.
printresult(test_result);
visualize(test_result, 1, strcat(algorithm, '-- k', num2str(algorithm_params.k)), algorithm_params);

%Comparison%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%algorithm_params.k = 6;
%test_result = runtest('algorithm3', algorithm_params, mdp_model, mdp, mdp_params, test_params);
%visualize(test_result, 1, 'algorithm3 -- k6', algorithm_params);

