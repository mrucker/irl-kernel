add_ourpaths; %includes all the paths required to run our code

%close all; clear; clc;
%global l1;
%global epsilon
%global lambda
%l1 = 1; 
%epsilon = 0.01;
%lambda = 1;

algorithm       = 'gpirl';
%algorithm       = 'maxent';
%algorithm       = 'mmpboost';
%algorithm       = 'learch';
%algorithm       = 'an';
algorithm       = 'algorithm3';
%algorithm       = 'mmp';
%algorithm       = 'firl';

%all_features = bad features. true_features = perfect features. (0,0) = good features.
algorithm_params = struct('all_features', 1, 'true_features',0 , 'epsilon', .0001, 'p', 128, 's', .01);
mdp_model        = 'standardmdp';%'linearmdp' (stochastic) or 'standardmdp' (deterministic)
mdp              = 'gridworld';%sum(100*clock)
mdp_params       = struct('n',10, 'determinism',1, 'seed',1000005, 'b',1, 'discount',.9);
test_params      = struct('training_sample_lengths', 5, 'training_samples', 10000, 'verbosity',0);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

%Visualize solution.
printresult(test_result);
visualize(test_result, 1, algorithm, algorithm_params);
