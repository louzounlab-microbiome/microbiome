from dafna.plot_rho import draw_rhos_calculation_figure
from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
import numpy as np
from infra_functions.general import apply_pca
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from collections import Counter


def prepare_data(tax_file, map_file, preform_z_scoring=True, taxnomy_level=6, n_components=20):
    OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, tax_file), os.path.join(SCRIPT_DIR, map_file), from_QIIME=False,
                         id_col='#OTU ID')

    preproccessed_data = preprocess_data(OtuMf.otu_file, preform_z_scoring=preform_z_scoring, visualize_data=True, taxnomy_level=taxnomy_level,
                                         preform_taxnomy_group=True)

    # otu_after_pca_wo_taxonomy, _, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=True)

    return OtuMf, preproccessed_data


def create_data_maps(mapping_file):
    samples_ids = mapping_file.index
    # ------------------------------------- Patient data -------------------------------------
    patient = mapping_file["Patient"]
    id_to_patient_map = {id: p for id, p in zip(samples_ids, patient)}
    patients = set(id_to_patient_map.values())
    patient_to_ids_map = {patient: [] for patient in patients}
    for i, patient in id_to_patient_map.items():
        patient_to_ids_map[patient].append(i)

    # ------------------------------------- Gain data -------------------------------------

    treat_gain = mapping_file["Treat_Gain"]
    gain_to_binary_map = {"N": 0, "Y": 1}
    id_to_treat_time_map = {sample: t.split(".")[0] for sample, t in zip(samples_ids, treat_gain) if str(t) != "nan"}
    id_to_treat_gain_map = {sample: gain_to_binary_map[t.split(".")[1]] for sample, t in zip(samples_ids, treat_gain) if str(t) != "nan"}

    time_to_ids_map = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for i, time in id_to_treat_time_map.items():
        time_to_ids_map[time].append(i)

    # ------------------------------------- BMI data -------------------------------------
    """
        BMI = mapping_file["BMI"]
    id_to_bmi_map = {id: bmi for id, bmi in zip(samples_ids, BMI)}  # if str(anti) != "nan"}  # avoid nan
    time_to_previous_time_map = {"A": "A", "B": "A", "C": "B"}
    id_to_delta_bmi_map = {}
    B_C_ids =time_to_ids_map["B"] + time_to_ids_map["C"]
    for current_id in B_C_ids:
        previous_id = str(id_to_patient_map[current_id]) + time_to_previous_time_map[id_to_treat_time_map[current_id]]
        if previous_id not in id_to_bmi_map.keys() or current_id not in id_to_bmi_map.keys():
            delta_bmi = np.nan
        else:
            delta_bmi = id_to_bmi_map[current_id] - id_to_bmi_map[previous_id]
        id_to_delta_bmi_map[current_id] = delta_bmi

    """
    BMI = mapping_file["BMI"]
    id_to_bmi_map = {id: bmi for id, bmi in zip(samples_ids, BMI)}  # if str(anti) != "nan"}  # avoid nan
    time_to_previous_time_map = {"C": "A"}
    id_to_delta_bmi_map = {}
    C_ids = time_to_ids_map["C"]
    for current_id in C_ids:
        previous_id = str(id_to_patient_map[current_id]) + time_to_previous_time_map[id_to_treat_time_map[current_id]]
        if previous_id not in id_to_bmi_map.keys() or current_id not in id_to_bmi_map.keys():
            delta_bmi = np.nan
        else:
            delta_bmi = id_to_bmi_map[current_id] - id_to_bmi_map[previous_id]
        B_id = str(id_to_patient_map[current_id]) + "B"
        id_to_delta_bmi_map[previous_id] = delta_bmi
        id_to_delta_bmi_map[B_id] = delta_bmi

    # ------------------------------------- antibiotics data -------------------------------------

    antibiotics = mapping_file["antibiotics"]
    id_to_antibiotics_map = {id: str(anti) for id, anti in zip(samples_ids, antibiotics)}  # if str(anti) != "nan"}  # avoid nan
    id_to_chemo_map = {id: str(anti) for id, anti in zip(samples_ids, antibiotics)}  # if str(anti) != "nan"}  # avoid nan

    # ------------------------------------- antibiotics and chemo time C data -------------------------------------

    # time C D E has no antibiotics and chemo info
    # create a map to tell us weather a patient received antibiotics and chemo in time A or B and use this info instead
    patient_to_antibiotics_time_A_B = {}
    for patient in patients:
        p_ids = patient_to_ids_map[patient]
        got_antibiotics = False
        for i in p_ids:
            if str(antibiotics[i]).startswith("yes"):
                got_antibiotics = True
                patient_to_antibiotics_time_A_B[patient] = 1
                break
            patient_to_antibiotics_time_A_B[patient] = 0

    patient_to_chemo_time_A_B = {}
    for patient in patients:
        p_ids = patient_to_ids_map[patient]
        got_chemo = False
        for i in p_ids:
            if str(antibiotics[i]).endswith("B"):
                got_chemo = True
                patient_to_chemo_time_A_B[patient] = 1
                break
            patient_to_chemo_time_A_B[patient] = 0


    for key, val in id_to_antibiotics_map.items():
        if val.startswith("no"):
            id_to_antibiotics_map[key] = 0
        elif val.startswith("yes"):
            id_to_antibiotics_map[key] = 1
        else:  # get patient
            p = id_to_patient_map[key]
            id_to_antibiotics_map[key] = patient_to_antibiotics_time_A_B[p]

        if val.endswith("A"):
            id_to_chemo_map[key] = 0
        elif val.endswith("B"):
            id_to_chemo_map[key] = 1
        else:  # get patient
            p = id_to_patient_map[key]
            id_to_chemo_map[key] = patient_to_chemo_time_A_B[p]

    # ------------------------------------- one-hot build up -------------------------------------

    # build one-hot representation -> [antibiotics?, chemo?]
    # predict:weight_gain_time_B?, weight_gain_time_C?
    time_point_to_ids_to_one_hots = {"A": {}, "B": {}, "C": {}}
    time_point_to_ids_to_bacteria = {"A": {}, "B": {}, "C": {}}
    for time_point in ["A", "B", "C"]:
        ids = time_to_ids_map[time_point]
        ids = [i for i in ids if i in id_to_antibiotics_map.keys() and i in id_to_chemo_map.keys()]
        # one_hots = [[id_to_antibiotics_map[i], id_to_chemo_map[i]] for i in ids]
        one_hots = [id_to_antibiotics_map[i] for i in ids]  # in time B C all patients got chemo
        for i, one in zip(ids, one_hots):
            time_point_to_ids_to_one_hots[time_point][i] = one
            time_point_to_ids_to_bacteria[time_point][i] = preproccessed_data.loc[i]

    return time_point_to_ids_to_one_hots, time_point_to_ids_to_bacteria, id_to_treat_time_map, id_to_delta_bmi_map, time_to_ids_map


def calc_linear_regression(X, y):
    X = np.array([X]).T
    y = np.array([y]).T
    reg = LinearRegression().fit(X, y)
    reg.score(X, y)
    b_1 = reg.coef_
    b_n = reg.intercept_

    # calculate the difference for each sample
    differences = []
    for x, real_y in zip(X, y):
        reg_y = np.dot(x, b_1) + b_n
        diff = real_y - reg_y
        differences.append(diff[0])
    return differences


if __name__ == "__main__":
    # tax_file = "taxonomy.csv"
    # map_file = 'mapping file with data Baniyahs Merge.csv'
    # OtuMf, preproccessed_data = prepare_data(tax_file, map_file)
    meta_data_file = "metadata_ok106ok115_chemo_merge.csv"
    table_file = "table.csv"
    tax_level = 5
    OtuMf, preproccessed_data = prepare_data(table_file, meta_data_file, preform_z_scoring=False, taxnomy_level=tax_level)
    # preproccessed_data = preproccessed_data.drop(["Unassigned"], axis=1)
    bacteria = preproccessed_data.columns
    otu_file = OtuMf.otu_file
    mapping_file = OtuMf.mapping_file
    # get [antibiotics, chemo] one hot for each sample, bacteria data, time and weight gain data
    time_point_to_ids_to_one_hots, time_point_to_ids_to_bacteria, id_to_treat_time_map, id_to_delta_bmi_map, time_to_ids_map\
        = create_data_maps(mapping_file)

    for time_point in ["A", "B"]:
        ids = time_to_ids_map[time_point]
        ids = [i for i in ids if i in id_to_delta_bmi_map.keys()]
        X = [time_point_to_ids_to_bacteria[time_point][i] for i in ids]
        y = [id_to_delta_bmi_map[i] for i in ids]
        # co-varient
        x1 = [time_point_to_ids_to_one_hots[time_point][i] for i in ids]

        nan_indexes = [i for i, val in enumerate(y) if str(val) == "nan"]
        [X, y, x1, ids] = pop_idx(nan_indexes, [X, y, x1, ids])
        differences = calc_linear_regression(x1, y)
        id_to_diff_map = {i: diff for i, diff in zip(ids, differences)}
        # look for correlations between differences and bacteria

        time_to_title_map = {"A": "pre-treatment", "B": "durin-treatment 9-12 weeks",
        "C": "4-6 months post treatment", "D": "6-12 months post treatment"}

        title = "Chemotherapy correlation task\nWeight gain during chemotherapy\nEliminating the effect of receiving antibiotics\n(time=" + time_to_title_map[time_point] + ")"
        draw_rhos_calculation_figure(id_to_diff_map, preproccessed_data, title, taxnomy_level=tax_level, num_of_mixtures=10, ids_list=ids,
                                     save_folder="correlation")

    print("!")
