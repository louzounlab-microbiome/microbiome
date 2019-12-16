function [otu_scale,X,F_hat_prime,growth_mean_p,growth_std_p,self_reg_mean_p,interact_std_p,deriv_std,perturb_std_p] = estMatricesHyperparameters(f,df_dt,intervene_matrix,rescale,perturbations)
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

% compute data matrices and estimate hyperparameters for MDSINE inference
% methods

% inputs:
% f = spline trajectory estimates; corresponds to f_hat in supplementary material (denoting that it is an *estimate*), but we omit this additional notation for cleanliness
% df_dt = spline derivative estimates; corresponds to f_hat_prime in supplementary material, but we use Leibniz notation here for clarity
% intervene_matrix = cell array of matrices indicating when OTUs are present/absent
% rescale = rescaling factor [usually set to 1; can change if there are numerical
% issues]
% perturbMask = (optional) indicates whether perturbations are active at each
% time-point

% outputs:
% X = matrix for solving the Lotka-Volterra equations
% F_hat_prime = cell of values of derivatives
% growth_mean_p = prior on growth parameter mean
% growth_std_p = prior on growth parameter std
% self_reg_mean_p = prior on self-regulation parameter mean
% interact_std_p = prior on self-regulation parameter standard deviation
% deriv_std = initial estimate of standard deviation of derivatives from gLV

numOTUs = size(f{1},2);
numSubjects = length(f);

X = cell(numOTUs,1);
F_hat_prime = cell(numOTUs,1);
% factor to re-scale OTU trajectories so values are standardized
otu_scale = ones(numOTUs,1)*rescale;

for o=1:numOTUs,
    X{o} = [];
    V2 = [];

    for i=1:numSubjects,
        no_intervene = find(intervene_matrix{i}(:,o) > 0);
        if ~isempty(no_intervene),
            MLVT = repmat(f{i}(no_intervene,o)/otu_scale(o),1,numOTUs);
            MLVT = MLVT.*f{i}(no_intervene,:)./repmat(otu_scale',length(no_intervene),1);
            if ~isempty(perturbations),
                perturbMask = perturbations{i};
                disp(i);
                disp(o)
                %MLVI = repmat(f{i}(no_intervene,o)/otu_scale(o),1,size(perturbMask,2)).*perturbMask;
                MLVI = repmat(f{i}(no_intervene,o)/otu_scale(o),1,size(perturbMask,2)).*perturbMask(no_intervene);
                X{o} = [X{o} ; [f{i}(no_intervene,o)/otu_scale(o) MLVT MLVI]];
            else,
                X{o} = [X{o} ; [f{i}(no_intervene,o)/otu_scale(o) MLVT]];
            end;
            V2 = [V2 ; df_dt{i}(no_intervene,o)/otu_scale(o)];
        end;
    end;
    F_hat_prime{o} = V2;
end;

for o=1:numOTUs,
    f = find(isnan(F_hat_prime{o}) | isinf(F_hat_prime{o}));
    F_hat_prime{o}(f) = [];
    X{o}(f,:) = [];
end;

a = zeros(numOTUs,2);

deriv_std = [];
perturb_std_p = [];
interact_std_p = [];
use_o = 1:numOTUs;
for i=1:numOTUs,
    Ato = pinv(X{i})*F_hat_prime{i};
    Ato(1) = abs(Ato(1));
    Ato(i+1) = -abs(Ato(i+1));
    a(i,1) = Ato(1);
    a(i,2) = Ato(i+1);
    deriv_std = [deriv_std ; (X{i}*Ato - F_hat_prime{i}).^2];
    interact_std_p = [interact_std_p a(i,2).^2];
    if ~isempty(perturbations),
        perturb_std_p = [perturb_std_p Ato((numOTUs+2):size(Ato,1)').^2];
    end;
end;
deriv_std = sqrt(mean(deriv_std));

interact_std_p = sqrt(mean(interact_std_p));
self_reg_mean_p = mean(-abs(a(use_o,2)));
growth_mean_p = mean(abs(a(use_o,1)));
growth_std_p = std(abs(a(use_o,1)));
if ~isempty(perturbations),
    perturb_std_p = sqrt(mean(perturb_std_p));
else,
    perturb_std_p = 0;
end;
