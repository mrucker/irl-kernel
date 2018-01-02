add_ourpaths; %includes all the paths required to run our code

%close all; clear; clc;
%global l1;
%global epsilon
%global lambda
%l1 = 1; 
%epsilon = 0.01;
%lambda = 1;

%algorithm       = 'gpirl';
%algorithm       = 'maxent';
%algorithm       = 'mmpboost';
%algorithm       = 'learch';
algorithm       = 'an';
algorithm       = 'algorithm3';
%algorithm       = 'algorithm5';
%algorithm       = 'mmp';
%algorithm       = 'firl';

%all_features = bad features. true_features = perfect features. (0,0) = good features.
algorithm_params = struct('all_features',1 , 'true_features',0 , 'epsilon', .0001, 'k',5, 'p', 3, 's',1, 'c', 1);
mdp_model        = 'linearmdp';%'linearmdp' (stochastic) or 'standardmdp' (deterministic)
mdp              = 'gridworld';%sum(100*clock)
mdp_params       = struct('n',4, 'determinism',1, 'seed',sum(100*clock), 'b',1, 'discount',.9);
test_params      = struct('training_sample_lengths', 2^2, 'training_samples', 2^10, 'verbosity',1);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

fprintf(1,'Seed:%10f\n', mdp_params.seed);

%Visualize solution.
printresult(test_result);
visualize(test_result, 1, strcat(algorithm, '-- k', num2str(algorithm_params.k)), algorithm_params);

%Comparison%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%algorithm_params.k = 6;
%test_result = runtest('algorithm3', algorithm_params, mdp_model, mdp, mdp_params, test_params);
%visualize(test_result, 1, 'algorithm3 -- k6', algorithm_params);

