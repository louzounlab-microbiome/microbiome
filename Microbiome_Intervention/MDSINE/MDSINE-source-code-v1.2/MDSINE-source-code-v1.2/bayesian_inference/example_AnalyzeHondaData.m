% do Bayesian inference on Honda dataset
load honda_data_processed_all.mat;
doBayesianInference('params_perturb/','output_perturb/',T,intervene_matrix,perturbations,densities,species_names,total_biomass,experimentBlocks);

% example of computing trajectories

% load processed data and MCMC samples
load output_perturb/bayes_inference.mat

% build initial condition vectors at day 3 from concentrations
Y0 = cell(7,1);
for s=1:7,
    Y0{s} = densities{s}(3,:);
end;

% calculate trajectories from lasso MCMC samples
[traj_lasso,traj_lasso_high,traj_lasso_low,percentSucceed_lasso] =  numIntTrajectoriesFromSamples(T,perturbations,experimentBlocks,intervene_matrix_filtered_merge,Y0,0,1,Theta_samples_lasso,[3 3],[65 29],[0.1 0.1],1,1);

% calculate trajectories from variable selection samples
[traj_select,traj_select_high,traj_select_low,percentSucceed_select] =  numIntTrajectoriesFromSamples(T,perturbations,experimentBlocks,intervene_matrix_filtered_merge,Y0,0,1,Theta_samples_select,[3 3],[65 29],[0.1 0.1],5,1);

% plot trajectories for firt three subjects
figure;
for s=1:3,
    if experimentBlocks(s) == 1,
        Ts = 3:0.1:65;
    else,
        Ts = 3:0.1:29;
    end;
    subplot(3,1,s);
    hold on;
    
    xlabel('time (days)');
    plot(Ts,traj_lasso{s});
    plot(T{s},densities{s});
    ylabel('ng DNA/g mL');
    title('Bayesian adaptive lasso');
end;

figure;
for s=1:3,
    if experimentBlocks(s) == 1,
        Ts = 3:0.1:65;
    else,
        Ts = 3:0.1:29;
    end;
    subplot(3,1,s);
    hold on;
    
    xlabel('time (days)');
    plot(Ts,traj_select{s});
    plot(T{s},densities{s});
    ylabel('ng DNA/g mL');
    title('Bayesian variable selection');
end;