import sys
import os
import math
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from LearningMethods.multi_model_learning import main

def main_pipeline():
    main_task = 'prognostic_PTSD_task_tax_level_'
    bactria_as_feature_file = '../sderot_anxiety/PTSD_data.csv'
    samples_data_file = '../sderot_anxiety/PTSD_tag.csv'
    rhos_folder = os.path.join('..', 'sderot_anxiety', 'rhos')
    pca_folder = os.path.join('..', 'sderot_anxiety', 'pca')

    taxonomy_range = [5,6]
    pca_range = range(2, 4)
    box_c_range = [pow(10, i) for i in range(-3, -1)]
    # parameters for preprocess
    for tax in taxonomy_range:
        for pca in pca_range:
                task = main_task + str(tax) + '_pca_' + str(pca)
                preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1,
                                   'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': '',
                                   'std_to_delete': 0, 'pca': pca}
                mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
                mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=False)
                mapping_file.rhos_and_pca_calculation(task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
                                                      rhos_folder, pca_folder)
                otu_path, mapping_path, pca_path = mapping_file.csv_to_learn(task, os.getcwd(), tax)

                # run svm learning
                folder = 'sderot_anxiety/'
                k_fold = 17
                test_size = 0.2
                names = ["no anxiety", "anxiety"]
                # get params dictionary from file / create it here
                p_dict = {"TASK_TITLE": task,
                          "FOLDER_TITLE": main_task,
                          "TAX_LEVEL": str(tax),
                          "CLASSES_NAMES": names,
                          "SVM": True,
                          "SVM_params": {'kernel': ['linear'],
                                         'gamma': ['scale'],
                                         'C': box_c_range,
                                         "create_coeff_plots": True,
                                         "CLASSES_NAMES": names,
                                         "K_FOLD": k_fold,
                                         "TEST_SIZE": test_size,
                                         "TASK_TITLE": task
                                         },
                          # if single option for each param -> single run, otherwise -> grid search.
                          "XGB": False,
                          "XGB_params": {},
                          # if single option for each param -> single run, otherwise -> grid search.
                          "NN": False,
                          "NN_params": {},  # if single option for each param -> single run, otherwise -> grid search.
                          "NNI": False,
                          "NNI_params": {},

                          # enter to model params?  might want to change for different models..
                          "K_FOLD": k_fold,
                          "TEST_SIZE": test_size,
                          #  ...... add whatever
                          }
                main(folder, otu_path, mapping_path, pca_path, p_dict)
                os.chdir('..')


if __name__ == '__main__':
    main_pipeline()
