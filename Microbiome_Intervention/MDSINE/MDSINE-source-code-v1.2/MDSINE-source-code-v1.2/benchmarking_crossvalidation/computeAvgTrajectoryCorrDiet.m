function [cc,rms] = computeAvgTrajectoryCorrDiet(concentration,Y_est,T_data,T_dense,T_test,otus)
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

% compute average correlation between actual and estimated trajectories

cc = 0;
rms = [];

T_data_idx = zeros(length(T_test),1);
T_dense_idx = zeros(length(T_test),1);
for t=1:length(T_test),
    f = find(T_data == T_test(t));
    T_data_idx(t) = f;

    f = find(T_dense == T_test(t));
    T_dense_idx(t) = f;
end;

for sidx=1:length(otus),
    s = otus(sidx);
    cor = corr(log10(concentration(T_data_idx,s)+0.1),log10(Y_est(T_dense_idx,s)+0.1));
    %cor = corr((concentration{s}(T_data_idx)),(Y_est(T_dense_idx,s)));
    cc = cc + cor(1,1);

    rms = [rms ; (log10(concentration(T_data_idx,s)+0.1) - log10(Y_est(T_dense_idx,s)+0.1)).^2];
    %rms = [rms ; ((concentration{s}(T_data_idx)) - (Y_est(T_dense_idx,s))).^2];
end;

cc = cc/length(otus);
rms = sqrt(sum(rms)/length(rms));
