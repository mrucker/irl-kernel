add_ourpaths; %includes all the paths required to run our code

%close all; clear; clc;
%global l1;
%global epsilon
%global lambda
%l1 = 1; 
%epsilon = 0.01;
%lambda = 1;

%algorithm       = 'firl';
%algorithm       = 'an';
algorithm        = 'algorithm2';
%all_features = bad features. true_features = perfect features. (0,0) = good features.
algorithm_params = struct('all_features',0,'true_features',1);
mdp_model        = 'linearmdp';
mdp              = 'gridworld';%sum(100*clock)
mdp_params       = struct('n',16, 'determinism',1, 'seed', 10, 'b',4, 'discount',.9);
test_params      = struct('training_sample_lengths', 128, 'training_samples', 512, 'verbosity',1);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

% Visualize solution.
printresult(test_result);
visualize(test_result, 1);
