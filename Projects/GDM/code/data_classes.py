import pandas as pd
from torch.utils.data import Dataset
ALL_FIELDS=0
FIELD_ONE_ONLY=1
FIELD_TWO_ONLY=2

class patient_samples_single_tp():
    def __init__(self, patient_dict):
        self.patient_dict = patient_dict
        self.id = patient_dict.get('ID', 'NO_ID')
        self.time_point = patient_dict.get('TP', 'NO_TP')
        self.field1 = patient_dict.get('FIELD0', None)
        self.field2 = patient_dict.get('FIELD1', None)
    def get_status(self):
        if self.field1 is not None and self.field2 is not None:
            return ALL_FIELDS
        elif self.field1 is not None:
            return FIELD_ONE_ONLY
        return FIELD_TWO_ONLY

class GDM_data_set(Dataset):
    def __init__(self, otu_df, mapping_table, patient_id_name, time_name, sample_type_name, transform=None):

        merged_df = pd.concat(
            [otu_df, mapping_table[patient_id_name], mapping_table[time_name], mapping_table[sample_type_name]], axis=1)
        patients_list = []
        unique_sample_types = list(sorted(merged_df[sample_type_name].unique()))
        for name, group in merged_df.groupby([patient_id_name, time_name]):
            patient_dict = {'ID': name[0], 'TP': name[1]}
            for i in range(len(group.index)):
                current_sample = group.iloc[i]
                current_microbiome_sample=current_sample.drop([sample_type_name,patient_id_name,time_name])
                if unique_sample_types.index(current_sample[sample_type_name]) == 0:
                    patient_dict['FIELD0'] = current_microbiome_sample
                else:
                    patient_dict['FIELD1'] = current_microbiome_sample
            patients_list.append(patient_samples_single_tp(patient_dict))
        self.patients = patients_list

        self.transform = transform

    def __len__(self):
        return len(self.patient_groups)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.patients[idx].patient_dict
        return self.transform(self.patients[idx]).patient_dict

    def separate_to_groups(self):
        indexes_of_patients_with_all_fields = [self.patients.index(patient) for patient in self.patients if patient.get_status()==ALL_FIELDS]
        indexes_of_patients_with_field1_only = [self.patients.index(patient) for patient in self.patients if patient.get_status()==FIELD_ONE_ONLY]
        indexes_of_patients_with_field2_only = [self.patients.index(patient) for patient in self.patients if patient.get_status()==FIELD_TWO_ONLY]
        return indexes_of_patients_with_all_fields, indexes_of_patients_with_field1_only, indexes_of_patients_with_field2_only
