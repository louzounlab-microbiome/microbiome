function [eigvals,fss,alpha_1,Beta_1]=...
    linstability_get_steadystates_and_eigenvalues(profile, Beta, alpha)
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

% Vanni Bucci
% for info:
% Assistant Professor
% Department of Biology
% University of Massachusetts Dartmouth
% 285 Old Westport Road
% N. Dartmouth, MA 02747-2300
% Phone: (508)999-9219
% Email: vbucci@umassd.edu
% Web: www.vannibucci.org
%-------------------
% get reduced matrix
Beta_1=Beta(profile==1,profile==1);
alpha_1=alpha(profile==1);

% -------------------
fss=zeros(length(alpha),1);
Beta_2=Beta(profile==1 & profile==1 , profile==1 & profile==1 );
alpha_2=alpha(profile==1 & profile==1);
fss(profile==1)=-inv(Beta_2)*alpha_2;
fss2=fss(profile==1);
Jacobian_matrix=zeros(size(Beta_1));
for im=1:size(Jacobian_matrix,1)
    for jm=1:size(Jacobian_matrix,2)
        if im==jm
            sum_ij=0;
            for km=1:size(Jacobian_matrix,2)
                sum_ij=sum_ij+Beta_1(im,km)*fss2(km);
            end
            Jacobian_matrix(im,jm)=alpha_1(im)+Beta_1(im,im)*fss2(im)+sum_ij;
        else
            Jacobian_matrix(im,jm)=Beta_1(im,jm)*fss2(im);
        end
    end
end
%
try
    eigvals=eig(Jacobian_matrix);
catch
    eigvals=NaN;
end

return
