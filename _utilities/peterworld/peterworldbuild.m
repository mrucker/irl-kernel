function [mdp_data,r,feature_data,true_feature_map] = peterworldbuild(mdp_params)


% Fill in default parameters.
mdp_params = peterworlddefaultparams(mdp_params);

% Set random seed.
rand('seed',mdp_params.seed);


% Build action mapping.
sa_s = zeros(12,86,12);
for s = 1:12
    sa_s(:,:,s) = s*ones(12,86);
end

sa_p = peterworldtransitions();



% Create MDP data structure.
mdp_data = struct(...
    'states',12,...
    'actions',86,...
    'discount',mdp_params.discount,...
    'determinism',1,...
    'sa_s',sa_s,...
    'sa_p',sa_p);

r = rand(12,1);

% Build the features.
[feature_data,true_feature_map] = peterworldfeatures(mdp_params,mdp_data);

end