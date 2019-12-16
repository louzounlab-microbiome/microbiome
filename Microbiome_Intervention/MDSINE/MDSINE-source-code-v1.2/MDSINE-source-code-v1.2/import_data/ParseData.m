function [all_data] = ParseData(metadata_file, counts_raw, biomass_raw, counts_dataFormat)
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


%% input:
%counts_raw = flat file of raw counts data for each OTU
%biomass_raw = flat file of biomass
%metadata = 7-col flat file denoting {sampleID, isIncluded, subjectID, measurementID,
%perturbation, intervention, expt_block}:
%there are #samples rows (ea. w/unique sampleID), sorted by
%[s1(t1),...,s1(tT),...,sN(t1),...,sN(tT)];
%isIncluded = bool col indicating whether or not the sample will be included in analysis (e.g., exclude ear/tail samples)
%perturbation = int col indicating the perturbation at that point (values
%{0,...,#perturbs});
%intervention = #OTU's-long vec specifying timepoint (index) at which ea.
%OTU is introduced (0 if it is not an "intervention"); fill in -1 (or any
%number <0) for the rest of the rows, or else with a tab character
%expt_block = int col indicating which experimental block the sample
%belongs to
%counts_dataFormat = int specifying the format of the counts input data; 1
%for BLASTN format (first row is species name, and each col is counts for
%each species; there may be a col of metadata/sampleID); 2 for qiime or
%mothur format (biom file standard)

%% output:
%struct all_data st.
%all_data.metadata = struct of metadata
%all_data.counts_data = struct of counts (#timepoints x #OTU's) for each
%subject
%all_data.biomass_data = struct containing the biomass data in CFU and CT
%for all mice, across all timepoints and replicates; first column of each
%matrix indicates the timepoint (each timepoint repeated for #replicates entries), and subsequent columns are the individual subjects

%% parse metadata *********************************************************
if(fopen(metadata_file)<3)
    error('Bad input metadata file.');
end

vars = tdfread(metadata_file,'\t');
all_fields = fieldnames(vars); %cell vec of strings for each fieldname
numFields = length(all_fields);
%keyboard;

sampleID = [];
isIncluded = [];
subjectID = [];
measurementID = [];

perturbID = [];
intervene_idx = [];
block_intv_vecs = []; %will be cell of vecs of intervention vecs for each experimental block

for i=1:length(all_fields)
    cur_name = lower(all_fields{i}(find(isstrprop(all_fields{i},'alpha'))));
    switch(cur_name)
        case('sampleid')
            sampleID = getfield(vars, all_fields{i});
        case('isincluded')
            isIncluded = getfield(vars, all_fields{i});
        case('subjectid') %N.B. this really encodes the "experimental unit"--i.e., if a subject belongs to different experimental blocks (e.g., crossover study design)
            ...they will need to be listed under different ID's, correspondingly; thus we always have #(experimental units) >= #subjects
            subjectID = getfield(vars, all_fields{i});
        case('measurementid')
            measurementID = getfield(vars, all_fields{i});
        case('perturbid')
            perturbID = getfield(vars, all_fields{i});
%        case ('intervention')
%            intervene_idx = getfield(vars, all_fields{i});
%            intervene_idx = intervene_idx(~isnan(intervene_idx));
        case('exptblock')
            exptBlockID = getfield(vars, all_fields{i});
        otherwise %intervention for each experimental block; each col is # OTU's long
            cur_intv_vec = getfield(vars, all_fields{i});
            block_intv_vecs = [block_intv_vecs; {cur_intv_vec(~isnan(cur_intv_vec))}];
    end
end

%keyboard;

if(isempty(sampleID))
    error('You must provide the sample ID corresponding to each measurement.');
end

if(isempty(isIncluded))
    error('You must indicate whether or not a sample (measurement) is to be included in analysis:  1/0 for incl./excl., resp.');
elseif(length(isIncluded)~=length(sampleID))
    error('The number of entries in your sample inclusion/exclusion data does not match your number of samples.');
end

if(isempty(subjectID))
    error('You must provide the subject ID corresponding to each measurement.');
elseif(length(subjectID)~=length(sampleID))
    error('The number of entries in your subject ID data does not match your number of samples.');
end

if(isempty(measurementID))
    error('You must provide the timepoint corresponding to each measurement.');
elseif(length(measurementID)~=length(sampleID))
    error('The number of entries in your timepoint data does not match your number of samples.');
end

if(isempty(exptBlockID))
    error('You must provide the experimental block identifier corresponding to each measurement.');
elseif(length(exptBlockID)~=length(sampleID))
    error('You must have exactly one experimental block identifier per sample.');
end

if(~isempty(perturbID) && length(perturbID)~=length(sampleID))
    error('The number of entries in your perturbation data does not match your number of samples.');
end


fclose('all');

% remove NAN (this is in case we have more species than samples)
sampleID(isnan(sampleID))=[];
isIncluded(isnan(isIncluded))=[];
subjectID(isnan(subjectID))=[];
measurementID(isnan(measurementID))=[];
exptBlockID(isnan(exptBlockID))=[];
perturbID(isnan(perturbID))=[];


incl_idx = find(isIncluded);
sampleID = sampleID(incl_idx);
subjectID = subjectID(incl_idx);
measurementID = measurementID(incl_idx);
if(~isempty(perturbID))
    perturbID = perturbID(incl_idx);
end
exptBlockID = exptBlockID(incl_idx);

numSamples = sum(isIncluded);

%numExptBlocks = length(full_time_vecs);

%map subjects to experimental blocks
i = 1;
subj_and_block = [subjectID(1), exptBlockID(1)];
while(i<length(subjectID))
    if(isempty(find(subj_and_block(:,1)==subjectID(i+1))))
        subj_and_block = [subj_and_block; [subjectID(i+1), exptBlockID(i+1)]];
    end
    i = i+1;
end

subj_blockID = subj_and_block(:,2);

%map time vectors to experimental blocks
uniqueBlocks = unique(exptBlockID);
numExptBlocks = length(uniqueBlocks);
block_time_vecs = cell(max(uniqueBlocks),1); %will be cell of vecs of time vecs for each experimental block
%expt blocks should be numbered 1:#blocks, else there will be empty cells
for i=1:numExptBlocks
   f = find(exptBlockID==uniqueBlocks(i));
   block_time_vecs{uniqueBlocks(i)} = unique(measurementID(f));
end

%keyboard;

numMeasurements = [];
cur_length = 0;

% subj_idx = subjectID(1);
% for i=1:numSamples
%     if(subjectID(i)==subj_idx)
%         cur_length = cur_length + 1;
%     else
%         numMeasurements = [numMeasurements; cur_length];
%         subj_idx = subjectID(i);
%         cur_length = 1;
%     end
% end
% numMeasurements = [numMeasurements; cur_length]; %last elem
uID = unique(subjectID);
for i=1:length(uID)
    cur_length=length(find(subjectID==uID(i)));
    numMeasurements = [numMeasurements; cur_length]; %last elem
end

numSubjects = length(numMeasurements);

%keyboard;

time_vecs = cell(numSubjects,1);

idx_offsets = [0; cumsum(numMeasurements(1:end-1))];
[tot_measurements,idx_max] = max(numMeasurements);

%for j=1:numSubjects
%    time_vecs{j} = measurementID(idx_offsets(j)+1:idx_offsets(j)+numMeasurements(j));
%end

for j=1:length(uID)
    time_vecs{j} = measurementID(find(subjectID==uID(j)));
end

%full_time_vec = time_vecs{idx_max};

perturb_vecs = cell(numSubjects,1);
if(~isempty(perturbID))
    perturb_vecs_0 = cell(numSubjects,1);
    %for j=1:numSubjects
    %    perturb_vecs_0{j} = perturbID(idx_offsets(j)+1:idx_offsets(j)+numMeasurements(j));
    %    perturb_vecs{j} = perturb_vecs_0{j};
    %end
    for j=1:length(uID)
        perturb_vecs_0{j} = perturbID(find(subjectID==uID(j)));
        perturb_vecs{j} = perturb_vecs_0{j};
    end
end


%% parse species counts data **********************************************
if(fopen(counts_raw)<3)
    error('Bad input counts file.');
end

concat_counts = []; %concatenated timeseries of counts across all subjects; each OTU will be a col

switch(counts_dataFormat)


    case(1) %BLASTN
        vars = tdfread(counts_raw,'\t');
        all_fields = fieldnames(vars);

        species_names = cell(length(all_fields));
        for i=1:length(all_fields)
           cur_name = lower(all_fields{i}(find(isstrprop(all_fields{i},'alpha'))));
            if(~strcmp(cur_name,'sampleid')) %case where their data contains some metadata ('SampleID') as a separate col
                concat_counts = [concat_counts, getfield(vars, all_fields{i})];
                species_names{i} = strtrim(all_fields{i});
            end
        end

        species_names = species_names(~cellfun('isempty',species_names));
        max_name_length = max(cellfun('length',species_names));

        fclose('all');

    case(2) %qiime or mothur biom standard file format

        OTU_table = readtable(counts_raw);
        species_names = cell(size(OTU_table,1),1);
        for i=1:size(OTU_table,1)
            cur_row_val = OTU_table{i,1};
            idx_1st_space = min(find(isspace(cur_row_val{1})));
            species_names{i} = cur_row_val{1}(1:idx_1st_space-1);
            cur_row_as_ints = sscanf(cur_row_val{1}(idx_1st_space+1:end),'%f');
            concat_counts = [concat_counts, cur_row_as_ints];
        end
        %species_names = species_names(k:end); %first entry may be parts of header comments, etc., or label of column; e.g., "OTU ID"
end

concat_counts = concat_counts(incl_idx,:); %filter out 'random' stuff like ear/tail measurements

species_names_char_array = [];
species_names = species_names(~cellfun('isempty',species_names));
max_name_length = max(cellfun('length',species_names));
for i=1:length(species_names)
    species_names_char_array = [species_names_char_array; [species_names{i},blanks(max_name_length - length(species_names{i}))]];
end


mouse_counts_0 = cell(numSubjects,1);
% for i=1:numSubjects
%     mouse_counts_0{i} = concat_counts(idx_offsets(i)+1:idx_offsets(i)+numMeasurements(i),:);
% end

for i=1:length(uID)
    mouse_counts_0{i} = concat_counts(find(subjectID==uID(i)),:);
end

numOTUs = size(concat_counts,2);

mouse_counts = cell(numSubjects,1);

mouse_counts = mouse_counts_0;


%% parse biomass data *****************************************************
if(fopen(biomass_raw)<3)
    error('Bad input biomass file.');
end

vars = tdfread(biomass_raw,'\t');
%vars = whos;
all_fields = fieldnames(vars);
numReplicates = length(all_fields);
%numReplicates = numFields - 3; %there are 3 other fields besides the col's reserved for replicates:  {{qPCR replicates}, DNA conc's, fecal weights, std curve coeff's}

%keyboard;

BMD_raw = cell(numSubjects,1);
all_rep_vals = cell(numSubjects,1);
for j=1:numReplicates
    all_rep_vals_j = getfield(vars, all_fields{j});%eval(vars(j).name);
    %for i=1:numSubjects
    %    all_rep_vals{i} = [all_rep_vals{i}, all_rep_vals_j(idx_offsets(i)+1:idx_offsets(i)+numMeasurements(i))];
    %end
    for i=1:length(uID)
        all_rep_vals{i} = [all_rep_vals{i}, all_rep_vals_j(find(subjectID==uID(i)))];
    end
end
for s=1:numSubjects
   length_T = length(all_rep_vals{s});
   BMD_raw{s} = zeros(numReplicates*length_T,1);
   for t=1:length_T
       BMD_raw{s}((t-1)*numReplicates+1:t*numReplicates) = all_rep_vals{s}(t,:)';
   end
end

fclose('all');


%% put biomass into form used by 'reformatBiomassData.m'
%% ^no...

%keyboard;



intervene_M = cell(numExptBlocks,1);
for e=1:numExptBlocks
    intervene_idx_cur = block_intv_vecs{e};%intervene_idx((e-1)*numOTUs+1:e*numOTUs);
    intervene_M{e} = ones(length(block_time_vecs{e}),numOTUs);
    for j=1:numOTUs
        if(intervene_idx_cur(j)>0)
            intervene_M{e}(1:intervene_idx_cur(j)-1,j) = 0; %by construction, intervene_idx will not have entries of 1
        end
    end
end

intervene_matrices = cell(numSubjects,1);

for s=1:numSubjects
    intervene_matrices{s} = intervene_M{subj_and_block(s,2)};
end

expt_block_T = cell(numSubjects,1);
for s=1:numSubjects
   expt_block_T{s} = block_time_vecs{subj_and_block(s,2)};
end

%keyboard;

subj_unique = unique(subjectID);
subj_fieldnames = cell(numSubjects,1);
for j=1:numSubjects
    subj_fieldnames{j} = strcat('s',num2str(subj_unique(j)));
end


perturbations = cell2struct(perturb_vecs,subj_fieldnames,1);
measurement_times = cell2struct(time_vecs,subj_fieldnames,1);
%exptBlocks = cell2struct(exptBlock_vecs,subj_fieldnames,1);

expt_block_timepoints = cell2struct(expt_block_T,subj_fieldnames,1);
intervene_matrices = cell2struct(intervene_matrices,subj_fieldnames,1);

metadata = struct('sampleID',sampleID,'subjectID',subjectID,'measurement_times',measurement_times,'perturbations',perturbations,'intervene_matrices',intervene_matrices,'subj_blockID',subj_blockID,'T',expt_block_timepoints,'species_names',species_names_char_array);
counts_data = cell2struct(mouse_counts,subj_fieldnames,1);
%biomass_data = struct('BMD_raw_CT',BMD_raw_CT,'BMD_raw_CFU',BMD_raw_CFU);
biomass_data = cell2struct(BMD_raw,subj_fieldnames,1);


all_data = struct('metadata',metadata,'counts_data',counts_data,'biomass_data',biomass_data);


%keyboard;

end