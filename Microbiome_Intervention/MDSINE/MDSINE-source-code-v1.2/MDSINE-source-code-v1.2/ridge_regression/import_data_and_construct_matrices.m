function [F,time,ID,U,taxa_data,time_data,ID_data,U_data,F_prime,X,magnitude]=...
    import_data_and_construct_matrices(densities,total_biomass,perturbations,time,o_rescale_to_one,...
    o_differentiation,specific_density,sample_indeces_to_remove);
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



% May 05 2015:
% Vanni Bucci, Ph.D.
% Assistant Professor
% Department of Biology
% Room: 335A
% University of Massachusetts Dartmouth
% 285 Old Westport Road
% N. Dartmouth, MA 02747-2300
% Phone: (508)999-9219
% Email: vbucci@umassd.edu
% Web: www.vannibucci.org


total_biomass_cat=[];
densities_cat=[];
perturbations_cat=[];
time_cat=[];
ID_cat=[];
% to loop through each sample
index_of_time_each_sample = [];

% Concatenate the trajectories
for i=1:length(densities)
    densities_cat=[densities_cat densities{i}'];
end

% Concatenate the perturbations
perturbations_cat=[];
if ~isempty(perturbations)
    for i=1:length(perturbations)
        perturbations_cat=[perturbations_cat perturbations{i}'];
    end
end

%Concatenate the total biomass
for i=1:length(total_biomass)
    total_biomass_cat=[total_biomass_cat total_biomass{i}'];
end

%Concatenate the time
for i=1:length(time)
    time_cat=[time_cat time{i}'];
    % For central difference
    index_of_time_each_sample = [index_of_time_each_sample size(time{i},1)];
end


%Concatenate the ID
for i=1:length(densities)
    ID_cat=[ID_cat repmat(i,1,size(densities{i},1))];
end

% Process the imported data
magnitude=floor(log10(mean(mean(total_biomass_cat)))); % magnitude of 16s (in DNA copies/g)

if o_rescale_to_one
	% The data are count-like and get normalized to one before being scaled by the total biomass (e.g. 16S)
	taxa_data=densities_cat.*repmat(total_biomass_cat*specific_density/(10^magnitude), ...
    	size(densities_cat,1), 1)./repmat(sum(densities_cat,1),size(densities_cat,1),1); % count-data matrix
else
	% The data are do not need to be scaled because they already represent some true ratio (qPCR)
	taxa_data=densities_cat.*repmat(total_biomass_cat*specific_density/(10^magnitude),...
        size(densities_cat,1), 1);
end

time_data=time_cat; % time
ID_data=ID_cat; % IDs
U_data=perturbations_cat;

% In case the perturbations are all zeros reduce the system
U_data(find(sum(U_data,2)==0),:)=[];
F=taxa_data;
time=time_data;
ID=ID_data;
U=U_data;

% Remove any sample that we do not want to use in the inference
% These indeces (if any) are from the input
if ~isempty(sample_indeces_to_remove)
    F(:,sample_indeces_to_remove)=[];
    U(:,sample_indeces_to_remove)=[];
    time(sample_indeces_to_remove)=[];
    ID(sample_indeces_to_remove)=[];
    taxa_data(:,sample_indeces_to_remove)=[];
    time_data(sample_indeces_to_remove)=[];
    U_data(:,sample_indeces_to_remove)=[];
    ID_data(sample_indeces_to_remove)=[];
end

if o_differentiation == 1 % If using forward difference (PLoS comp bio version)
    % Set Up the inference Matrices F,Y,U
    F_prime=diff(log(F),1,2)./repmat(diff(time),[size(F,1) 1]);
    % discard last data point of every sample/ID
    I=find(diff(time)<=0); % identify indices of last data points
    F_prime(:,I)=[];
    F(:,[I end])=[];
    time(:,[I end])=[];
    if ~isempty(U)
        U(:,[I end])=[];
    end
    ID(:,[I end])=[];

    % remove NaN/Inf/-INf :
    % - NaN happens when x(tk)=x(tk+1)=0
    % - Inf happens when x(tk)=0 and x(tk+1)!=0
    % - -Inf happens when x(tk)!=0 and x(tk+1)=0
    F_prime(isnan(F_prime))=0; % conditions where x(tk)=x(tk+1)=0 are set to 0
    [ix,iy]=find(isinf(F_prime));
    for i = 1:length(ix)
        s=sign(F_prime(ix(i),iy(i)));
        if s==-1
            F_prime(ix(i),iy(i))=-1/(time(iy(i)+1)-time(iy(i)));
        else
            F_prime(ix(i),iy(i))=0; % here means that x(tk)=0
        end
    end



elseif o_differentiation ==  2 % Using central difference in approximation of first derivative

    % f'(x) = ( f(x(t(k+1))) - f(x(t(k-1))) )/( (t(k+1)) - (t(k-1)) )
    % At first point for each sample,  using forward difference
    % At the last point for each sample, using backward difference
    F_prime = log(F);

    F_prime_copy = F_prime;

    first_column_index_sample = 1; % first column of each sample
    final_column_index_sample = 0; % last column of each sample
    for i=1:size(index_of_time_each_sample,2)


        for j=2:index_of_time_each_sample(i)-1

            column_index = j +final_column_index_sample ;
            % Central Difference
            F_prime_copy(:,column_index) = (F_prime(:,column_index+1) - F_prime(:,column_index-1) )/(time(column_index+1)- time(column_index-1)) ;

        end
        %First point using forward difference
         F_prime_copy (:,first_column_index_sample) =  (F_prime(:,first_column_index_sample+1) - F_prime(:,first_column_index_sample))/(time(first_column_index_sample+1) - time(first_column_index_sample));
        first_column_index_sample = first_column_index_sample + index_of_time_each_sample(i);

        %Last Point using backward difference
        final_column_index_sample =  final_column_index_sample + index_of_time_each_sample(i);
        F_prime_copy (:,final_column_index_sample) =  (F_prime(:,final_column_index_sample) - F_prime(:,final_column_index_sample -1))/(time(final_column_index_sample) - time(final_column_index_sample -1));

    end

    F_prime = F_prime_copy;
    %remove Inf/-Inf for Backward and Forward Difference at the end points
    %and starting points respectively

    first_column_index_sample = 1;
    final_column_index_sample = 0;
    for i=1:size(index_of_time_each_sample,2)

        final_column_index_sample = final_column_index_sample + index_of_time_each_sample(i);
        for j= 1:size(F_prime,1)

            %remove Inf/-Inf for Backward Difference at the end points
            if isinf(F_prime(j,final_column_index_sample))
                s = sign(F_prime(j,final_column_index_sample));
                if s==-1
                    F_prime(j,final_column_index_sample)=0; % here means that x(tk)=0
                else
                    F_prime(j,final_column_index_sample)=1/(time(final_column_index_sample)-time(final_column_index_sample-1)); %x(tk -1) = 0
                end
            end
            %remove Inf/-Inf for Forward Difference at the starting points
            if isinf(F_prime(j,first_column_index_sample))
                s = sign(F_prime(j,first_column_index_sample));
                if s==-1
                    F_prime(j,first_column_index_sample)=-1/(time(first_column_index_sample+1)-time(first_column_index_sample));% x(tk+1) = 0
                else
                    F_prime(j,first_column_index_sample)=0; % here means that x(tk)=0
                end
            end
        end
        first_column_index_sample = first_column_index_sample + index_of_time_each_sample(i);
    end


    % remove NaN/Inf/-INf for central diff code:
    % - NaN happens when x(tk-1)=x(tk+1)=0
    % - Inf happens when x(tk-1)=0 and x(tk+1)!=0
    % - -Inf happens when x(tk-1)!=0 and x(tk+1)=0
    %
    F_prime(isnan(F_prime))=0; % conditions where x(tk-1)=x(tk+1)=0 are set to 0
    [ix,iy]=find(isinf(F_prime));
    for i = 1:length(ix)
        s=sign(F_prime(ix(i),iy(i)));

        if F(ix(i),iy(i))~= 0 % if x(tk) != 0

             if s==-1
                 F_prime(ix(i),iy(i))=-F(ix(i),iy(i)-1)/(F(ix(i),iy(i))*(time(iy(i)+1)-time(iy(i)-1)));

             else
                 F_prime(ix(i),iy(i))=F(ix(i),iy(i)+1)/(F(ix(i),iy(i))*(time(iy(i)+1)-time(iy(i)-1)));


             end

        else
             F_prime(ix(i),iy(i))=0; %  x(tk)=0
        end
    end




 elseif o_differentiation ==  3 % Using forward difference in approximation of first derivative
    % Forward Difference
    % f'(x) = ( f(x(t(k+1))) - f(x(t(k))) )/( (t(k+1)) - (t(k)) )
    % At the last point, using backward difference
    % f'(x) = ( f(x(t(k))) - f(x(t(k-1))) )/( (t(k)) - (t(k-1)) )
    F_prime = log(F);
    F_prime_copy = F_prime;

    final_column_sample = 0; % final column of each sample
    for i=1:size(index_of_time_each_sample,2)

        for j=1:index_of_time_each_sample(i)-1 % To use forward difference except the end point

            column_index = j +final_column_sample ;
            F_prime_copy(:,column_index) = (F_prime(:,column_index+1) - F_prime(:,column_index) )/(time(column_index+1)- time(column_index)) ;

        end
        final_column_sample = final_column_sample + index_of_time_each_sample(i);

        %Using Backward Difference for the end points
        t1 = time(final_column_sample) - time(final_column_sample-1);

        F_prime_copy(:,final_column_sample) = (F_prime(:,final_column_sample) - F_prime(:,final_column_sample-1) )/t1;

    end

    F_prime = F_prime_copy;

    %remove Inf/-Inf for Backward Difference at the end points of
    %different samples

    column_index = 0;
    for i=1:size(index_of_time_each_sample,2)

        column_index = column_index + index_of_time_each_sample(i);
        for j= 1:size(F_prime,1)

            if isinf(F_prime(j,column_index))
                s = sign(F_prime(j,column_index));
                if s==-1
                    F_prime(j,column_index)=0; % here means that x(tk)=0
                else
                    F_prime(j,column_index)=1/(time(column_index)-time(column_index-1)); %x(tk -1) = 0
                end
            end
        end

    end

    % remove NaN/Inf/-INf for forward difference :
    % - NaN happens when x(tk)=x(tk+1)=0
    % - Inf happens when x(tk)=0 and x(tk+1)!=0
    % - -Inf happens when x(tk)!=0 and x(tk+1)=0
    F_prime(isnan(F_prime))=0; % conditions where x(tk)=x(tk+1)=0 are set to 0
    [ix,iy]=find(isinf(F_prime));
    for i = 1:length(ix)
        s=sign(F_prime(ix(i),iy(i)));
        if s==-1
            F_prime(ix(i),iy(i))=-1/(time(iy(i)+1)-time(iy(i)));
        else
            F_prime(ix(i),iy(i))=0; % here means that x(tk)=0
        end
    end


else
    error('No good differentiation option set');
end





% make the X matrix
if ~isempty(U)
    X=[ones(1,size(F,2));F;U];
else
    X=[ones(1,size(F,2));F];
end

return
