from Projects.GVHD_BAR.load_merge_otu_mf import OtuMfHandler
from Preprocess.preprocess import preprocess_data
import warnings
from Preprocess.general import apply_pca
import pandas as pd
import datetime
import os
import warnings

warnings.filterwarnings("ignore")
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

def prepare_data(n_components=20, preform_z_scoring=True, taxnomy_level=6):
    OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, 'saliva_samples_231018.csv'),
                         os.path.join(SCRIPT_DIR,
                                      'saliva_samples_mapping_file_231018.csv'), from_QIIME=True)
    preproccessed_data = preprocess_data(OtuMf.otu_file, preform_z_scoring, visualize_data=True, taxnomy_level=taxnomy_level,
                                         preform_taxnomy_group=True)
    otu_after_pca_wo_taxonomy, _, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=True)

    ######## Pre process (Remove control group) ########
    OtuMf.mapping_file['DATE_datetime'] = OtuMf.mapping_file['DATE'].apply(get_datetime)
    OtuMf.mapping_file['Mocosities_start_datetime'] = OtuMf.mapping_file['Mucositis_Start'].apply(get_datetime)
    OtuMf.mapping_file['TIME_BEFORE_MOCO_START'] = OtuMf.mapping_file['Mocosities_start_datetime'] - OtuMf.mapping_file[
        'DATE_datetime']

    OtuMf.mapping_file['time_for_the_event'] = OtuMf.mapping_file['TIME_BEFORE_MOCO_START'].apply(get_days)

    OtuMf.mapping_file['time_for_the_event'][
        OtuMf.mapping_file['Mocosities_start_datetime'] == datetime.datetime.strptime('01/01/1900', '%d/%m/%Y')] = 9999
    # create groups
    data_grouped = OtuMf.mapping_file.groupby('Personal_ID')
    censored_data = {}
    not_censored = pd.DataFrame()
    dilated_df = pd.DataFrame()
    y_for_deep = pd.DataFrame()
    x_for_deep = pd.DataFrame()
    x_for_deep_censored = pd.DataFrame()
    y_for_deep_censored = pd.DataFrame()

    for subject_id, subject_data in data_grouped:
        if 9999 in subject_data['time_for_the_event'].values:  # censored
            tmp_data = subject_data.join(otu_after_pca_wo_taxonomy)
            tmp_data_only_valid = tmp_data.loc[tmp_data[0].notnull()]
            if not tmp_data_only_valid.empty:
                x_for_deep_censored = x_for_deep_censored.append(subject_data)

                tmp_data_only_valid['time_before_moco_start_days'] = tmp_data_only_valid[
                    'TIME_BEFORE_MOCO_START'].apply(get_days)
                tmp_data_only_valid.sort_index(by='time_before_moco_start_days', ascending=False, inplace=True)
                tmp_data_only_valid['relative_start_date'] = tmp_data_only_valid['time_before_moco_start_days'].iloc[
                                                                 0] - tmp_data_only_valid[
                                                                 'time_before_moco_start_days']
                tmp_data_only_valid['relative_max_date'] = tmp_data_only_valid['relative_start_date'][-1] - \
                                                           tmp_data_only_valid['relative_start_date']
                tmp_data_only_valid['delta_time'] = -1
                tmp_data_only_valid['mse_coeff'] = 0
                tmp_data_only_valid['time_sense_coeff'] = 1
                y_for_deep_censored = y_for_deep_censored.append(
                    tmp_data_only_valid[['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

                # get only the last sample
                censored_data[subject_id] = tmp_data_only_valid.loc[
                    tmp_data_only_valid['TIME_BEFORE_MOCO_START'] == min(tmp_data_only_valid['TIME_BEFORE_MOCO_START'])]

        else:  # not censored
            before_event_mask = subject_data['time_for_the_event'] > 0
            before_event_subjects = subject_data.loc[before_event_mask]
            if not before_event_subjects.empty:
                not_censored = not_censored.append(before_event_subjects)
                dilated_df = dilated_df.append(before_event_subjects)

                x_for_deep = x_for_deep.append(before_event_subjects)
                before_event_subjects['time_before_moco_start_days'] = before_event_subjects[
                    'TIME_BEFORE_MOCO_START'].apply(get_days)
                before_event_subjects.sort_index(by='time_before_moco_start_days', ascending=False, inplace=True)
                before_event_subjects['relative_start_date'] = before_event_subjects['time_before_moco_start_days'].iloc[
                                                                   0] - before_event_subjects['time_before_moco_start_days']
                before_event_subjects['relative_max_date'] = before_event_subjects['relative_start_date'] + \
                                                             before_event_subjects['time_before_moco_start_days']
                before_event_subjects['delta_time'] = before_event_subjects['time_for_the_event']
                before_event_subjects['mse_coeff'] = 1
                before_event_subjects['time_sense_coeff'] = 0
                y_for_deep = y_for_deep.append(
                    before_event_subjects[['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

    x_for_deep = x_for_deep.join(otu_after_pca_wo_taxonomy)
    x_for_deep = x_for_deep.loc[x_for_deep[0].notnull()]
    y_for_deep = y_for_deep.loc[x_for_deep.index]

    x_for_deep_censored = x_for_deep_censored.join(otu_after_pca_wo_taxonomy)
    x_for_deep_censored = x_for_deep_censored.loc[x_for_deep_censored[0].notnull()]
    y_for_deep_censored = y_for_deep_censored.loc[x_for_deep_censored.index]

    return x_for_deep, y_for_deep, x_for_deep_censored, y_for_deep_censored, censored_data, not_censored,\
           otu_after_pca_wo_taxonomy, OtuMf, preproccessed_data

"""
from Projects.GVHD_BAR.load_merge_otu_mf import OtuMfHandler
from Preprocess.Preprocess import preprocess_data
import warnings
from Preprocess.general import apply_pca
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
    OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, 'saliva_samples_231018.csv'),
                         os.path.join(SCRIPT_DIR,
                                      'saliva_samples_mapping_file_231018.csv'), from_QIIME=True)
    preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=True, taxnomy_level=6,
                                         preform_taxnomy_group=True)
    otu_after_pca_wo_taxonomy, _, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=True)

    ######## Pre process (Remove control group) ########
    OtuMf.mapping_file['DATE_datetime'] = OtuMf.mapping_file['DATE'].apply(get_datetime)
    OtuMf.mapping_file['Mocosities_start_datetime'] = OtuMf.mapping_file['Mucositis_Start'].apply(get_datetime)
    OtuMf.mapping_file['TIME_BEFORE_MOCO_START'] = OtuMf.mapping_file['Mocosities_start_datetime'] - OtuMf.mapping_file[
        'DATE_datetime']

    OtuMf.mapping_file['time_for_the_event'] = OtuMf.mapping_file['TIME_BEFORE_MOCO_START'].apply(get_days)

    OtuMf.mapping_file['time_for_the_event'][
        OtuMf.mapping_file['Mocosities_start_datetime'] == datetime.datetime.strptime('01/01/1900', '%d/%m/%Y')] = 9999
    # create groups
    data_grouped = OtuMf.mapping_file.groupby('Personal_ID')
    censored_data = {}
    not_censored = pd.DataFrame()
    dilated_df = pd.DataFrame()
    y_for_deep = pd.DataFrame()
    x_for_deep = pd.DataFrame()
    x_for_deep_censored = pd.DataFrame()
    y_for_deep_censored = pd.DataFrame()

    for subject_id, subject_data in data_grouped:
        if 9999 in subject_data['time_for_the_event'].values:  # censored
            tmp_data = subject_data.join(otu_after_pca_wo_taxonomy)
            tmp_data_only_valid = tmp_data.loc[tmp_data[0].notnull()]
            if not tmp_data_only_valid.empty:
                x_for_deep_censored = x_for_deep_censored.append(subject_data)

                tmp_data_only_valid['time_before_moco_start_days'] = tmp_data_only_valid[
                    'TIME_BEFORE_MOCO_START'].apply(get_days)
                tmp_data_only_valid.sort_index(by='time_before_moco_start_days', ascending=False, inplace=True)
                tmp_data_only_valid['relative_start_date'] = tmp_data_only_valid['time_before_moco_start_days'].iloc[
                                                                 0] - tmp_data_only_valid[
                                                                 'time_before_moco_start_days']
                tmp_data_only_valid['relative_max_date'] = tmp_data_only_valid['relative_start_date'][-1] - \
                                                           tmp_data_only_valid['relative_start_date']
                tmp_data_only_valid['delta_time'] = -1
                tmp_data_only_valid['mse_coeff'] = 0
                tmp_data_only_valid['time_sense_coeff'] = 1
                y_for_deep_censored = y_for_deep_censored.append(
                    tmp_data_only_valid[['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

                # get only the last sample
                censored_data[subject_id] = tmp_data_only_valid.loc[
                    tmp_data_only_valid['TIME_BEFORE_MOCO_START'] == min(tmp_data_only_valid['TIME_BEFORE_MOCO_START'])]

        else:  # not censored
            before_event_mask = subject_data['time_for_the_event'] > 0
            before_event_subjects = subject_data.loc[before_event_mask]
            if not before_event_subjects.empty:
                not_censored = not_censored.append(before_event_subjects)
                dilated_df = dilated_df.append(before_event_subjects)

                x_for_deep = x_for_deep.append(before_event_subjects)
                before_event_subjects['time_before_moco_start_days'] = before_event_subjects[
                    'TIME_BEFORE_MOCO_START'].apply(get_days)
                before_event_subjects.sort_index(by='time_before_moco_start_days', ascending=False, inplace=True)
                before_event_subjects['relative_start_date'] = before_event_subjects['time_before_moco_start_days'].iloc[
                                                                   0] - before_event_subjects['time_before_moco_start_days']
                before_event_subjects['relative_max_date'] = before_event_subjects['relative_start_date'] + \
                                                             before_event_subjects['time_before_moco_start_days']
                before_event_subjects['delta_time'] = before_event_subjects['time_for_the_event']
                before_event_subjects['mse_coeff'] = 1
                before_event_subjects['time_sense_coeff'] = 0
                y_for_deep = y_for_deep.append(
                    before_event_subjects[['relative_start_date', 'delta_time', 'relative_max_date', 'mse_coeff', 'time_sense_coeff']])

    x_for_deep = x_for_deep.join(otu_after_pca_wo_taxonomy)
    x_for_deep = x_for_deep.loc[x_for_deep[0].notnull()]
    y_for_deep = y_for_deep.loc[x_for_deep.index]

    x_for_deep_censored = x_for_deep_censored.join(otu_after_pca_wo_taxonomy)
    x_for_deep_censored = x_for_deep_censored.loc[x_for_deep_censored[0].notnull()]
    y_for_deep_censored = y_for_deep_censored.loc[x_for_deep_censored.index]

    return x_for_deep, y_for_deep, x_for_deep_censored, y_for_deep_censored, censored_data, not_censored, otu_after_pca_wo_taxonomy, OtuMf
"""