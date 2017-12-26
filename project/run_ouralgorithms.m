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
algorithm_params = struct('all_features', 0, 'true_features',1 , 'epsilon', .000001, 'p', 128, 's', .01);
mdp_model        = 'linearmdp';%'linearmdp' (stochastic) or 'standardmdp' (deterministic)
mdp              = 'gridworld';%sum(100*clock)
mdp_params       = struct('n',64, 'determinism',1, 'seed',1000006, 'b',4, 'discount',.8);
test_params      = struct('training_sample_lengths', 128, 'training_samples', 2^9, 'verbosity',1);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

%Visualize solution.
printresult(test_result);
visualize(test_result, 1, algorithm, algorithm_params);
