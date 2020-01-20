import os
import pandas as pd
import numpy as np
from LearningMethods.multi_model_learning import multi_model_learning_main
from infra_functions.preprocess_loop_helper import microbiome_preprocess, microbiome_preprocess_evaluation, \
    extra_features_preprocess, extra_features_preprocess_evaluation, fill_and_normalize_extra_features, create_tags_csv, \
    create_na_distribution_csv


def anxiety_main(csv_path, calc_na_distribution=True, create_tags=False, calc_rhos=True):
    df = pd.read_csv(csv_path)
    ids = df.index
    id_col_name = "Barcode"
    ptsd_time_5_tag = "T5_PTSD"
    life_ptsd_time_5_tag = "PTSD_throughoutlife"
    internalize_time_5_tag = "T5_Internalizing"
    life_internalize_time_5_tag = "internalizing_throughoutlife"
    c_or_m = "CorM"
    expose_or_control = "Exposure"
    if create_tags:
        create_tags_csv(df, id_col_name, [(c_or_m, "child_or_mother"), (expose_or_control, "expose_or_control")])

        create_child_tags(df, c_or_m, [(ptsd_time_5_tag, "ptsd_time_5"),
                                       (life_ptsd_time_5_tag, "life_ptsd"),
                                       (internalize_time_5_tag, "internalize_time_5"),
                                       (life_internalize_time_5_tag, "life_internalize")])

    # ptsd_time_5_tag, life_ptsd_time_5_tag, internalize_time_5_tag, life_internalize_time_5_tag

    medical_questionnaires_and_psychological_diagnoses_cols = df.columns[29:89]
    df = df.drop(columns=medical_questionnaires_and_psychological_diagnoses_cols)

    df[c_or_m] = [val.upper() for val in df[c_or_m]]
    child_df = df[df[c_or_m] == "C"].replace(" ", np.nan)  # x samples
    mother_df = df[df[c_or_m] == "M"].replace(" ", np.nan)  # x samples

    control_child_df = child_df[child_df[expose_or_control] == 0].replace(" ", np.nan)  # 109 samples
    control_mother_df = mother_df[mother_df[expose_or_control] == 0].replace(" ", np.nan)  # 108 samples
    exposed_child_df = child_df[child_df[expose_or_control] == 1].replace(" ", np.nan)  # 123 samples
    exposed_mother_df = mother_df[mother_df[expose_or_control] == 1].replace(" ", np.nan)  # 121 samples
    c_m_names = ["control_child_df", "exposed_child_df",
                 "control_mother_df", "exposed_mother_df"]

    if calc_na_distribution:
        create_na_distribution_csv(df,
                                   [control_child_df, exposed_child_df,
                                   control_mother_df, exposed_mother_df],
                                   c_m_names,
                                   "Mother_or_child_and_Exposure")


def create_child_tags(df, c_or_m, tag_col_and_name_list):
    for tag, name in tag_col_and_name_list:
        df = df[df[c_or_m] != "M"]
        df = df[df[c_or_m] != "m"]
        create_csv_from_column(df, tag, "child_" + name + "_tag.csv")


def extra_features_for_child(df_path):
    na_df = pd.read_csv("Mother_or_child_and_Exposere_feature_na_distribution.csv")

    na_df = na_df.set_index("column_name")
    na_df = na_df[["control_child_df", "exposed_child_df"]]
    na_df = na_df.drop(na_df.index[0:27])
    selected_features = []
    for feature, control_val, exposed_val in zip(na_df.index, na_df["control_child_df"], na_df["exposed_child_df"]):
        if control_val < 0.25 and exposed_val < 0.25:
            selected_features.append(feature)
    selected_features.remove("PrimaryLast")
    selected_features.remove("filter_$")
    df = pd.read_csv(df_path)
    df = df.set_index("Barcode")
    extra_features_df = df[selected_features]
    extra_features_df = fill_and_normalize_extra_features(extra_features_df)
    extra_features_df.to_csv("extra_features.csv")
    return extra_features_df


def anxiety_learning(project_folder_and_task, task_name, classes_names, tax, pca):
    otu_path = os.path.join(project_folder_and_task, 'OTU_merged_' + task_name + '_task_tax_level_' + tax + '_pca_' + pca + '.csv')
    mapping_path = os.path.join(project_folder_and_task, 'Tag_file_' + task_name + '_task.csv')
    pca_path = os.path.join(project_folder_and_task, 'Pca_obj_' + task_name + '_task_pca_' + pca + '.pkl')
    k_fold = 17
    test_size = 0.2
    # get params dictionary from file / create it here
    dict = {"TASK_TITLE": task_name,  # the name of the task for plots titles...
            "FOLDER_TITLE": project_folder_and_task,  # creates the folder for the task we want to do, save results in it
            "TAX_LEVEL": str(tax),
            "CLASSES_NAMES": classes_names,
            "SVM": True,
            "SVM_params": {'kernel': ['linear'],
                           'gamma': ['auto'],
                           'C': [0.01, 0.1, 1, 10, 100, 1000],
                           "create_coeff_plots": True,
                           "CLASSES_NAMES": classes_names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": task_name
                           },
            # if single option for each param -> single run, otherwise -> grid search.
            "XGB": False,
            "XGB_params": {'learning_rate': [0.1],
                           'objective': ['binary:logistic'],
                           'n_estimators': [1000],
                           'max_depth': [7],
                           'min_child_weight': [1],
                           'gamma': [1],
                           "create_coeff_plots": True,
                           "CLASSES_NAMES": classes_names,
                           "K_FOLD": k_fold,
                           "TEST_SIZE": test_size,
                           "TASK_TITLE": "sderot_anxiety"
                           },  # if single option for each param -> single run, otherwise -> grid search.
            "NN": False,
            "NN_params": {
                "hid_dim_0": 120,
                "hid_dim_1": 160,
                "reg": 0.68,
                "lr": 0.001,
                "test_size": 0.1,
                "batch_size": 32,
                "shuffle": 1,
                "num_workers": 4,
                "epochs": 150,
                "optimizer": 'SGD',
                "loss": 'MSE',
                "model": 'tanh_b'
            },  # if single option for each param -> single run, otherwise -> grid search.
            "NNI": False,
            "NNI_params": {
                "result_type": 'auc'
            },
            # enter to model params?  might want to change for different models..
            "K_FOLD": k_fold,
            "TEST_SIZE": test_size,
            #  ...... add whatever
            }
    multi_model_learning_main(project_folder_and_task, otu_path, mapping_path, pca_path, dict)


if __name__ == "__main__":
    # create tag files for learning...
    csv_path = "Microbiom_Karen_T5_9.1.2020.csv"
    run_main = False
    if run_main:
        anxiety_main(csv_path)

    tag_list = ["child_ptsd_life", "child_ptsd_5", "child_internalize_life", "child_internalize_5"]
    pca_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    tax_list = [5, 6]

    microbime_selection = False
    if microbime_selection:
        microbiome_preprocess(pca_list, tax_list, tag_list)

    microbime_evaluation = False
    if microbime_evaluation:
        microbiome_preprocess_evaluation(tag_options=tag_list, pca_options=pca_list, tax_options=tax_list)

    pca_list = [1, 2, 3, 4, 5, 6, 7]
    folder = "extra_features_files"
    df_path = "extra_features.csv"
    results_path = "preprocess_evaluation_plots"
    id_col_name = "Barcode"

    feature_selection = False
    if feature_selection:
        selected_features = extra_features_for_child(csv_path)

        extra_features_preprocess(pca_list, tag_list, id_col_name, folder, df_path, results_path)

    feature_evaluation = False
    if feature_evaluation:
        extra_features_preprocess_evaluation(folder, tag_list, pca_list, results_path)

    learning = False
    if learning:
        task = 'internalize_5' #'expose_or_control' # 'child_or_mother' # 'internalize_life' # ptsd_  #  internalize_
        tax = '5'
        pca = '9'
        project_folder_and_task = task + '_tax_' + tax + '_csv_files'
        names = ["no internalizing disorder", "internalizing disorder"]  # ["control", "exposed"]  # ["child", "mother"]  #["no anxiety", "anxiety"]  #
        anxiety_learning(project_folder_and_task, task, names, tax, pca)


    """
    metadata = pd.read_csv("ok121_orna_metadata.csv", sep="\t")
    id_to_code_map = {int(id): code for id, code in zip(metadata["SampleID"][:133], metadata["Sample_code"][:133])}
    taxon_names = pd.read_csv("taxonomy.csv", sep="\t")
    tax_hash_to_name_map = {hash: name for hash, name in zip(taxon_names["Feature ID"], taxon_names["Taxon"])}
    otu = pd.read_csv("ok121_orna_otu_table.csv", sep="\t")
    otu.columns = ["ID"] + [id_to_code_map[col] for col in otu.columns[1:].astype(int)]
    otu["Taxonomy"] = [tax_hash_to_name_map[hash[1:-1]] if hash[1:-1] in tax_hash_to_name_map.keys() else np.nan for hash in otu["ID"]]
    otu.columns = [c[1:-1] for c in otu.columns]
    otu["ID"] = [c[1:-1] for c in otu["ID"]]
    otu = otu.set_index('ID')
    otu = otu.T
    otu.to_csv("otu.csv")
   
    otu = pd.read_csv("otu.csv")
    otu.columns = [c[1:-1] for c in otu.columns]
    otu["ID"] = [c[1:-1] for c in otu["ID"]]
    # otu = otu.rename(columns={'Unnamed: 0': "ID"})
    otu = otu.set_index('ID')
    otu.index = [id.replace("S", "F") for id in otu.index]
    otu.to_csv("otu.csv")
    tag = pd.read_csv("ptsd_5_tag.csv")
    tag = tag.set_index('ID')
    tag.to_csv("tag.csv")
    
    ids_otu = list(set(otu.index[:-1]))
    ids_tag = list(tag.index)
    ids = [id for id in ids_tag if id in ids_otu]
    import collections
    print([item for item, count in collections.Counter(ids_otu).items() if count > 1])
    
    
    c = otu.loc["A5945FC"]
    len(set(c.iloc[0]))
    len(set(c.iloc[1]))
    otu = otu.drop(["A5945FC"])
    otu.loc["A5945FC"] = c.iloc[0]
    
    c = otu.loc["A5977FM"]
    len(set(c.iloc[0]))
    len(set(c.iloc[1]))
    otu = otu.drop(["A5977FM"])
    otu.loc["A5977FM"] = c.iloc[0]
    
    c = otu.loc["A5919FC"]
    len(set(c.iloc[0]))
    len(set(c.iloc[1]))
    otu = otu.drop(["A5919FC"])
    otu.loc["A5919FC"] = c.iloc[1]
    
    d = otu.loc["taxonomy"]
    otu = otu.drop(["taxonomy"])
    otu.loc["taxonomy"] = d
    """

