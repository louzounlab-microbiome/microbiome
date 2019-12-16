% do Bayesian inference on C. diff dataset
load cdiff_data_reformatted.mat;
doBayesianInference('params_cdiff/','output_cdiff/',T,intervene_matrix,[],data_counts,species_names,BMD,experimentBlocks);

% example of computing trajectories for C. diff from day 30 to day 56

% load processed data and MCMC samples
load output_cdiff/bayes_inference.mat

% calculate concentrations from counts data and raw biomass data
[concentrations] = calcConcentrations(T,data,BMD,intervene_matrix,3,keep_species);
% build initial condition vectors at day 30 from concentrations
[Y0] = buildInitialConditionVectorsFromData(T,30*ones(5,1),concentrations);

% calculate trajectories from lasso MCMC samples
[traj_lasso,traj_lasso_high,traj_lasso_low,percentSucceed_lasso] = numIntTrajectoriesFromSamples(T,[],experimentBlocks,intervene_matrix_filtered_merge,Y0,med_biomass,med_counts,Theta_Samples_lasso,30,56,0.1,1,1);
% calculate trajectories from variable selection samples
[traj_select,traj_select_high,traj_select_low,percentSucceed_select] = numIntTrajectoriesFromSamples(T,[],experimentBlocks,intervene_matrix_filtered_merge,Y0,med_biomass,med_counts,Theta_Samples_select,30,56,0.1,1,1);

% plot trajectories
figure;
for s=1:5,
    subplot(5,1,s);
    hold on;
    
    xlabel('time (days)');
    plot(30:0.1:56,log(traj_lasso{s}(:,2)),'-b','LineWidth',3);
    plot(30:0.1:56,log(traj_select{s}(:,2)),'-r','LineWidth',3);
    plot(T{s}(16:26),log(concentrations{s,2}(16:26)),'k-*');
    ylabel('log CFU/g mL');
    
    %plot(30:0.1:56,traj_lasso{s}(:,2),'-b','LineWidth',3);
    %plot(30:0.1:56,traj_select{s}(:,2),'-r','LineWidth',3);
    %plot(T(16:26),concentrations{s,2}(16:26),'k-*');
    %ylabel('CFU/g mL');
    
    if s==1,
        legend('lasso','select','data');
    end;
end;