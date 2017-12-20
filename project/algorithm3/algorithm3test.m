add_ourpaths; %includes all the paths required to run our code

close all; clear; clc;

algorithm_params = struct('p', 2);
mdp_model        = 'linearmdp';
mdp              = 'gridworld';
mdp_params       = struct('n',2,'determinism',1,'seed', sum(100*clock), 'b',1, 'discount',.9);
test_params      = struct('training_sample_lengths', 128, 'training_samples', 64, 'verbosity',1);
verbosity        = test_params.verbosity;

[mdp_data,r,feature_data,true_feature_map] = feval(strcat(mdp,'build'),mdp_params);

test_params = setdefaulttestparams(test_params);
mdp_solution = feval(strcat(mdp_model,'solve'),mdp_data,r);

if isempty(test_params.true_examples)
    example_samples = sampleexamples(mdp_model,mdp_data,mdp_solution,test_params);
else
    example_samples = test_params.true_examples;
end

irl_result = algorithm2run(algorithm_params, mdp_data, mdp_model, feature_data, example_samples, true_feature_map, verbosity);

