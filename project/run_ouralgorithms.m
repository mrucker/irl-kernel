add_ourpaths; %includes all the paths required to run our code

%close all; clear; clc;
%global l1;
%global epsilon
%global lambda
%l1 = 1; 
%epsilon = 0.01;
%lambda = 1;

%algorithm        = 'gpirl';
algorithm       = 'firl';
algorithm       = 'an';
algorithm       = 'algorithm3';
%algorithm       = 'mmp';
%algorithm       = 'mmpboost';

%all_features = bad features. true_features = perfect features. (0,0) = good features.
algorithm_params = struct('all_features', 1, 'true_features',0 , 'epsilon', .01, 'p', 2, 's', .3);
mdp_model        = 'linearmdp';
mdp              = 'gridworld';
mdp_params       = struct('n',32, 'determinism',1, 'seed', sum(100*clock), 'b',1, 'discount',.9);
test_params      = struct('training_sample_lengths', 100, 'training_samples', 128, 'verbosity',0);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

% Visualize solution.
printresult(test_result);
visualize(test_result, 1, algorithm, algorithm_params);
