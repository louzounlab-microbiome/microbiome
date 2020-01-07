import os

from Microbiome_Intervention.Bifidobacterium_bifidum.bifidobacterium_bifidum_data_set import \
    Bifidobacterium_bifidum_DataLoader
from Microbiome_Intervention.Diet_study.diet_study_data_set import Diet_Study_DataLoader
from Microbiome_Intervention.MDSINE_data_cdiff.cdiff_data_set import CdiffDataLoader
from Microbiome_Intervention.MDSINE_data_diet.mdsine_diet_study_data_set import MDSINE_Study_DataLoader
from Microbiome_Intervention.MITRE_data_bokulich.bokulich_data_set import BokulichDataLoader
from Microbiome_Intervention.MITRE_data_david.david_data_set import DavidDataLoader
from Microbiome_Intervention.VitamineA.vitamin_a_data_set import VitaminADataLoader
from Microbiome_Intervention.significant_bacteria import check_if_bacteria_correlation_is_significant, \
    get_significant_beta_from_file


def sign_bact(results_path):
    # if we want to run on a single model, change to a different csv file with the same structure
    if calc_significant_bacteria:
        folder = os.path.join("tax=" + str(tax), "Significant_bacteria")
        for algo in ["ard regression"]:
            results_df_path = os.path.join("tax=" + str(tax), results_path)
            significant_bacteria = check_if_bacteria_correlation_is_significant(results_df_path, algo)
            get_significant_beta_from_file(results_df_path, algo, significant_bacteria, folder)

"""
def Bifidobacterium_bifidum(tax, create_data, created_X_y_files, calc_significant_bacteria, cross_validation, results_path):
    title = 'Bifidobacterium_bifidum'
    bactria_as_feature_file = os.path.join('Bifidobacterium_bifidum', 'otu_table_P2_4times.csv')
    samples_data_file = os.path.join('Bifidobacterium_bifidum', 'Fasting_map2.csv')

    if create_data:
        bifidum_dataset = Bifidobacterium_bifidum_DataLoader(title=title,
                                                             bactria_as_feature_file=bactria_as_feature_file,
                                                             samples_data_file=samples_data_file, taxnomy_level=tax,
                                                             created_data=created_X_y_files,
                                                             cross_validation=cross_validation, test_size=test_size)

    # if we want to run on a single model, change to a different csv file with the same structure
    if calc_significant_bacteria:
        sign_bact(results_path)

"""

def Diet_study(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, results_path):
    data = 'Diet_study'
    bactria_as_feature_file = 'taxonomy_counts_all_together.csv'
    samples_data_file = 'SampleID_map.csv'

    diet_dataset = Diet_Study_DataLoader(data_name=data, bactria_as_feature_file=bactria_as_feature_file,
                                    samples_data_file=samples_data_file, taxnomy_level=tax,
                                    create_regression_data=create_regression_data,
                                    create_time_serie_data=create_time_serie_data)
    diet_dataset.run(run_regression, run_rnn, run_lstm, cross_validation=cross_validation, test_size=test_size)

    if calc_significant_bacteria:
        sign_bact(results_path)


def MDSINE_data_cdiff(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, results_path):
    data = 'MDSINE_data_cdiff'
    bactria_as_feature_file = 'counts.csv'
    samples_data_file = 'metadata.csv'
    tax = 7

    cdiff_dataset = CdiffDataLoader(data_name=data, bactria_as_feature_file=bactria_as_feature_file,
                                    samples_data_file=samples_data_file, taxnomy_level=tax,
                                    create_regression_data=create_regression_data,
                                    create_time_serie_data=create_time_serie_data)

    cdiff_dataset.run(run_regression, run_rnn, run_lstm, cross_validation=cross_validation, test_size=test_size)

    if calc_significant_bacteria:
        sign_bact(results_path)


def MDSINE_data_diet(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, results_path):
    data = 'MDSINE_data_diet'
    bactria_as_feature_file = 'counts.csv'
    samples_data_file = 'metadata.csv'
    tax = 7

    diet_dataset = MDSINE_Study_DataLoader(data_name=data, bactria_as_feature_file=bactria_as_feature_file,
                                    samples_data_file=samples_data_file, taxnomy_level=tax,
                                    create_regression_data=create_regression_data,
                                    create_time_serie_data=create_time_serie_data)
    diet_dataset.run(run_regression, run_rnn, run_lstm, cross_validation=cross_validation, test_size=test_size)

    if calc_significant_bacteria:
        sign_bact(results_path)


def MITRE_data_bokulich(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, results_path):
    data = 'MITRE_data_bokulich'
    bactria_as_feature_file = 'dafna_proccessed_abundance.csv'
    samples_data_file = 'sample_metadata_no_repeats.csv'

    bokulich_dataset = BokulichDataLoader(data_name=data, bactria_as_feature_file=bactria_as_feature_file,
                                samples_data_file=samples_data_file, taxnomy_level=tax,
                                create_regression_data=create_regression_data,
                                create_time_serie_data=create_time_serie_data)
    bokulich_dataset.run(run_regression, run_rnn, run_lstm, cross_validation=cross_validation, test_size=test_size)

    if calc_significant_bacteria:
        sign_bact(results_path)


def MITRE_data_david(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, results_path):
    data = 'MITRE_data_david'
    bactria_as_feature_file = 'dafna_proccessed_abundance.csv'
    samples_data_file = 'sample_metadata.csv'

    david_dataset = DavidDataLoader(data_name=data, bactria_as_feature_file=bactria_as_feature_file,
                                    samples_data_file=samples_data_file, taxnomy_level=tax,
                                    create_regression_data=create_regression_data,
                                    create_time_serie_data=create_time_serie_data)

    david_dataset.run(run_regression, run_rnn, run_lstm, cross_validation=cross_validation, test_size=test_size)

    if calc_significant_bacteria:
        sign_bact(results_path)


def VitamineA(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, results_path):
    data = 'VitamineA'
    bactria_as_feature_file = 'ok16_va_otu_table.csv'
    samples_data_file = 'metadata_ok16_va.csv'

    vitamin_dataset = VitaminADataLoader(data_name=data, bactria_as_feature_file=bactria_as_feature_file,
                                samples_data_file=samples_data_file, taxnomy_level=tax,
                                create_regression_data=create_regression_data,
                                create_time_serie_data=create_time_serie_data)
    vitamin_dataset.run(run_regression, run_rnn, run_lstm, cross_validation=cross_validation, test_size=test_size)

    if calc_significant_bacteria:
        sign_bact(results_path)


if __name__ == "__main__":
    data_sets_names = ['MDSINE_data_cdiff', 'MDSINE_data_diet', 'Diet_study',
                       'MITRE_data_bokulich', 'MITRE_data_david', 'VitamineA']  # 'MDSINE_data_cdiff']
    tax = 5
    create_data_loader = True
    calc_significant_bacteria = False
    test_size = 0.5
    cross_validation = 10  # 5  # "loo"
    create_regression_data = False
    create_time_serie_data = True
    run_regression = False
    run_rnn = False
    run_lstm = False
    # as the wanted parameter in preform_learning
    all_times_all_bact_results_path = os.path.join("tax=" + str(tax), str(cross_validation) + "_fold_test_size_" + str(test_size) + "_results_df.csv")
    # all_times_all_bact_results_path = "all_times_all_bacteria_best_models_results_df.csv"

    # for tax in [5, 6]:
    MDSINE_data_cdiff(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, all_times_all_bact_results_path)
    
    MDSINE_data_diet(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, all_times_all_bact_results_path)

    Diet_study(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, all_times_all_bact_results_path)
    MITRE_data_bokulich(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, all_times_all_bact_results_path)
    MITRE_data_david(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, all_times_all_bact_results_path)
    VitamineA(tax, create_regression_data, create_time_serie_data,
                      run_regression, run_rnn, run_lstm,
                      cross_validation, test_size,
                      calc_significant_bacteria, all_times_all_bact_results_path)





