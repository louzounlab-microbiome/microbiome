%% calling wrapper for numerical integration (C. diff simulations)

pathogen_ID = 2; %C. diff currently hardcoded elsewhere as OTU #2
pathogen_t0 = 1; %pathogen introduction time
numSubjects = length(T_sample);
experimentBlocks = ones(numSubjects,1);

med_biomass = 22.3358; %these values obtained from a run (with fewer MCMC samples) 
med_counts = 5.7898*10^4;

for i=1:length(sims)
[traj] = wrap_numIntTrajFromSamples(T_sample,sims(i).tstart,sims(i).tend,sims(i).dt,[],experimentBlocks,pathogen_ID,pathogen_t0,sims(i).initialState,sims(i).Density_toIntroduce,med_biomass,med_counts,sims(i).glv_pars)

end