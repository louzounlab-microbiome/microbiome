function [f,f_high,f_low,df_dt,df_dt_high,df_dt_low] = estOTUTrajDerivFromSplineSamples(phi_sample,T,B,BD,intervene_matrix,scaleFactor)
% This file is a part of the MDSINE program.
%
% MDSINE: Microbial Dynamical Systems INference Engine
% Infers dynamical systems models from microbiome time-series datasets, and
% predicts biologically relevant behaviors of the ecosystems.
% Copyright (C) 2015 Vanni Bucci, Georg Gerber
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% estimate noise free trajectories and derivatives from MCMC spline samples

% f = trajectory estimates (f_high = 97.5% CI estimates, f_low = 2.5% CI
% estimates)

% df_dt = derivative estimates (df_dt_high = 97.5% CI estimates, df_dt_low = 2.5% CI
% estimates)

numSubjects = size(phi_sample,1);
numOTUs = size(phi_sample,2);

f = cell(numSubjects,1);
f_high = cell(numSubjects,1);
f_low = cell(numSubjects,1);
df_dt = cell(numSubjects,1);
df_dt_high = cell(numSubjects,1);
df_dt_low = cell(numSubjects,1);

for s=1:numSubjects,
    f{s} = zeros(length(T),numOTUs);
    f_high{s} = zeros(length(T),numOTUs);
    f_low{s} = zeros(length(T),numOTUs);
    df_dt{s} = zeros(length(T),numOTUs);
    df_dt_high{s} = zeros(length(T),numOTUs);
    df_dt_low{s} = zeros(length(T),numOTUs);
    for o=1:numOTUs,
        uv = find(intervene_matrix(:,o) > 0);
        enus = exp(B{o}*phi_sample{s,o}+scaleFactor{s,o});
        edir = BD{o}*phi_sample{s,o};

        f{s}(uv,o) = median(enus,2);
        f_high{s}(uv,o) = prctile(enus,97.5,2);
        f_low{s}(uv,o) = prctile(enus,2.5,2);

        enus = enus.*edir;
        df_dt{s}(uv,o) = median(enus,2);
        df_dt_high{s}(uv,o) = prctile(enus,97.5,2);
        df_dt_low{s}(uv,o) = prctile(enus,2.5,2);
    end;
end;


