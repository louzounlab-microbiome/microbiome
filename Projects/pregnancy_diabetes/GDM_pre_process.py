import sys
import os
import math
import itertools
from joblib import Parallel, delayed
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from GDM_run_pipeline import learning_pipeline



def main_preprocess(tax, site, trimester, preprocess_prms):
    main_task = 'GDM_taxonomy_level_' + str(tax)+'_'+site+'_trimester_'+trimester
    bactria_as_feature_file = 'GDM_OTU_rmv_dup.csv'
    
    samples_data_file = 'GDM_tag_rmv_dup_' +trimester + '_' + site +'.csv'
    rhos_folder = os.path.join('pregnancy_diabetes_'+trimester+'_'+site, 'rhos')
    pca_folder = os.path.join('pregnancy_diabetes_'+trimester+'_'+site, 'pca')

    mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
    mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=False)
    mapping_file.rhos_and_pca_calculation(main_task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
                                         rhos_folder, pca_folder)
    otu_path, mapping_path, pca_path = mapping_file.csv_to_learn(main_task, os.path.join(os.getcwd(), 'pregnancy_diabetes_'+trimester+'_'+site), tax)
    return otu_path, mapping_path, pca_path


def parallel_pipeline(t):
    trimester = t[0]
    site = t[1]
    tax = int(t[2])
    
    # parameters for Preprocess
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1,
                     'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': '',
                     'std_to_delete': 0.2, 'pca': 0}

    otu_path, mapping_path, pca_path = main_preprocess(tax, site, trimester, preprocess_prms)
    learning_pipeline(otu_path, mapping_path, pca_path, tax, site, trimester)


if __name__ == '__main__':
    s = [['T1'], ['STOOL'], [5]]
    arg_list=list(itertools.product(*s))
    for arg in arg_list:
        parallel_pipeline(arg)
    #Parallel(n_jobs=8)(delayed(parallel_pipeline)(arg) for arg in arg_list)