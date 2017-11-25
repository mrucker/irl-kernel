addpath(genpath(fullfile(fileparts(which(mfilename)),'../_dependencies/')));
%close all; clear; clc;
global l1;
global epsilon
global lambda
l1 = 1;
epsilon = 0.01;
lambda = 1;

algorithm        = 'firl';
%algorithm        = 'gpirl';
algorithm_params = struct();
mdp_model        = 'linearmdp';
mdp              = 'gridworld';
mdp_params       = struct('n',32,'determinism',1,'seed', sum(100*clock), 'b',4, 'discount',0.9);
test_params      = struct('training_sample_lengths', 32, 'training_samples', 512, 'verbosity',2);

test_result = runtest(algorithm, algorithm_params, mdp_model, mdp, mdp_params, test_params);

% Visualize solution.
printresult(test_result);
visualize(test_result, 1);
