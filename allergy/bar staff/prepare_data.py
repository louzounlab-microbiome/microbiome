from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
import warnings

from infra_functions.general import apply_pca
import pandas as pd


import datetime

import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))




def get_datetime(date_str, time_format='%d/%m/%Y'):
    if pd.isnull(date_str):
        date_str = '01/01/1900'
        return datetime.datetime.strptime(date_str, '%d/%m/%Y')
    try:
        return datetime.datetime.strptime(date_str, time_format)
    except ValueError:
        try:
            # use %Y
            time_format = time_format.split('/')
            time_format[-1] = '%Y'
            time_format = '/'.join(time_format)
            return datetime.datetime.strptime(date_str, time_format)
        except ValueError:
            try:
                # use %y
                time_format = time_format.split('/')
                time_format[-1] = '%y'
                time_format = '/'.join(time_format)
                return datetime.datetime.strptime(date_str, time_format)
            except:
                warnings.warn(f'{date_str} is not a valid date, sample will be ignored')
                date_str = '01/01/1800'
                return datetime.datetime.strptime(date_str, '%d/%m/%Y')


def get_days(days_datetime):
    return days_datetime.days

def prepare_data(n_components = 20):
    OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'),
                         os.path.join(SCRIPT_DIR,
                                      'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'),
                         from_QIIME=True, id_col='Feature ID', taxonomy_col='Taxonomy')
    preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=True, taxnomy_level=5, taxonomy_col='Taxonomy',
                                         preform_taxnomy_group=True)
    otu_after_pca_wo_taxonomy, _, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=True)

    ######## Pre process (Remove control group) ########
    column_to_use_for_filter = 'AllergyTypeData131118'
    OtuMf.mapping_file = OtuMf.mapping_file.loc[OtuMf.mapping_file['AllergyTypeData131118'] != 'Con']

    ######## get date of sample in date format ########
    date_of_sample_col = 'Date'
    OtuMf.mapping_file['Date_of_sample'] = OtuMf.mapping_file[date_of_sample_col].apply(get_datetime,
                                                                                        time_format='%m/%d/%y')

    ######## remove invalid subjects (those who had samples with no dates or bad dates) ########
    # bad dates
    tmp = OtuMf.mapping_file.loc[OtuMf.mapping_file['Date_of_sample'].isin(['1800-01-01', '1900-01-01'])]
    patients_with_bad_date = tmp['PatientNumber210119'].unique()
    # remove bad dates
    OtuMf.mapping_file = OtuMf.mapping_file.loc[~OtuMf.mapping_file['PatientNumber210119'].isin(patients_with_bad_date)]

    ######## Calculate time for event ########
    OtuMf.mapping_file['time_for_the_event'] = 9999
    col_to_group_by = 'PatientNumber210119'
    data_grouped = OtuMf.mapping_file.groupby(col_to_group_by)

    for subject_id, subject_data in data_grouped:
        if any(subject_data['SuccessDescription'] == 'A1'):  # Uncensored
            date_of_event = subject_data['Date_of_sample'].max()
            time_for_the_event = date_of_event - subject_data['Date_of_sample']
            tmp_df = OtuMf.mapping_file.loc[subject_data.index]
            tmp_df['time_for_the_event'] = time_for_the_event.apply(get_days)
            OtuMf.mapping_file.update(tmp_df)
        else:  # Censored
            pass

    ######## Filter alergies ########
    # allergy types ['Sesame', 'Peanut', 'Egg', 'Non', 'Walnut', 'Milk', 'Cashew', 'Hazelnut']
    # OtuMf.mapping_file['AllergyTypeData131118'].value_counts()
    # Peanut    134
    # Milk    112
    # Sesame    80
    # Walnut    72
    # Egg    28
    # Cashew    18
    # Hazelnut    9
    # Non    9
    allergy_to_use = ['Peanut']
    OtuMf.mapping_file = OtuMf.mapping_file[OtuMf.mapping_file['AllergyTypeData131118'].isin(allergy_to_use)]

    ######## Create inputs ########

    # create groups
    col_to_group_by = 'PatientNumber210119'
    data_grouped = OtuMf.mapping_file.groupby(col_to_group_by)
    censored_data = {}
    not_censored = pd.DataFrame()
    y_for_deep = pd.DataFrame()
    x_for_deep = pd.DataFrame()
    x_for_deep_censored = pd.DataFrame()
    y_for_deep_censored = pd.DataFrame()

    def calculate_y_for_deep_per_row(row):
        a = row.sort_values()
        return a.index[0]

    for subject_id, subject_data in data_grouped:
        if 9999 in subject_data['time_for_the_event'].values:  # censored
            tmp_data = subject_data.join(otu_after_pca_wo_taxonomy)
            tmp_data_only_valid = tmp_data.loc[tmp_data[0].notnull()]
            if not tmp_data_only_valid.empty:
                x_for_deep_censored = x_for_deep_censored.append(subject_data)

                tmp_data_only_valid.sort_index(by='Date_of_sample', ascending=True, inplace=True)
                tmp_data_only_valid['relative_start_date'] = (
                            tmp_data_only_valid['Date_of_sample'] - tmp_data_only_valid['Date_of_sample'].iloc[
                        0]).apply(get_days)
                tmp_data_only_valid['relative_max_date'] = (
                            tmp_data_only_valid['Date_of_sample'].iloc[-1] - tmp_data_only_valid[
                        'Date_of_sample']).apply(get_days)
                tmp_data_only_valid['delta_time'] = -1
                tmp_data_only_valid['mse_coeff'] = 0
                tmp_data_only_valid['time_sense_coeff'] = 1
                y_for_deep_censored = y_for_deep_censored.append(
                    tmp_data_only_valid[
                        ['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

                # get only the last sample
                censored_data[subject_id] = tmp_data_only_valid.loc[
                    tmp_data_only_valid['relative_max_date'] == min(tmp_data_only_valid['relative_max_date'])]

        else:  # not censored
            before_event_mask = subject_data['time_for_the_event'] > 0
            before_event_subjects = subject_data.loc[before_event_mask]
            if not before_event_subjects.empty:
                not_censored = not_censored.append(before_event_subjects)

                x_for_deep = x_for_deep.append(before_event_subjects)
                before_event_subjects.sort_index(by='time_for_the_event', ascending=False, inplace=True)
                before_event_subjects['relative_start_date'] = before_event_subjects['time_for_the_event'].iloc[
                                                                   0] - before_event_subjects['time_for_the_event']
                before_event_subjects['relative_max_date'] = before_event_subjects['time_for_the_event']
                before_event_subjects['delta_time'] = before_event_subjects['time_for_the_event']
                before_event_subjects['mse_coeff'] = 1
                before_event_subjects['time_sense_coeff'] = 0
                y_for_deep = y_for_deep.append(
                    before_event_subjects[
                        ['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

    x_for_deep = x_for_deep.join(otu_after_pca_wo_taxonomy)
    x_for_deep = x_for_deep.loc[x_for_deep[0].notnull()]
    y_for_deep = y_for_deep.loc[x_for_deep.index]

    x_for_deep_censored = x_for_deep_censored.join(otu_after_pca_wo_taxonomy)
    x_for_deep_censored = x_for_deep_censored.loc[x_for_deep_censored[0].notnull()]
    y_for_deep_censored = y_for_deep_censored.loc[x_for_deep_censored.index]

    return x_for_deep, y_for_deep, x_for_deep_censored, y_for_deep_censored, censored_data, not_censored, otu_after_pca_wo_taxonomy, OtuMf