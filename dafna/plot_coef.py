from os.path import join
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from dafna.general_functions import shorten_bact_names


def create_coeff_plots_by_alogorithm(averages, bacterias, task_name, algorithm, num_of_iters, edge_percent=4, folder=False):
    min_rho = min(averages)
    max_rho = max(averages)
    rho_range = max_rho - min_rho
    # we want to take those who are located on the sides of most (center 98%) of the mixed tags entries
    # there for the bound isn't fixed, and is dependent on the distribution of the mixed tags

    # lower_bound = min_rho + (rho_range * 0.3)
    # upper_bound = max_rho - (rho_range * 0.3)
    upper_bound = np.percentile(averages, 100 - edge_percent)
    lower_bound = np.percentile(averages, edge_percent)
    significant_bacteria_and_rhos = []

    for i, bact in enumerate(bacterias):
        if averages[i] < lower_bound or averages[i] > upper_bound:  # significant
            significant_bacteria_and_rhos.append([bact, averages[i]])

    significant_bacteria_and_rhos.sort(key=lambda s: s[1])

    if folder:
        if not os.path.exists(folder):
            os.makedirs(folder)
        result_csv = join(folder, algorithm + "_" + task_name + "_significant_bacteria_coeff_average_of_" +
                               str(num_of_iters) + "_runs.csv")
    else:
        result_csv = algorithm + "_" + task_name + "_significant_bacteria_coeff_average_of_" + str(num_of_iters) + "_runs.csv"

    if significant_bacteria_and_rhos:
        df = pd.DataFrame(significant_bacteria_and_rhos)
        df.columns = ["bacteria", "rhos"]
        df.to_csv(result_csv, index=False)
    else:  # no significant_bacteria
        df = pd.DataFrame([["no significant_bacteria", "0"]])
        df.columns = ["bacteria", "rhos"]
        df.to_csv(result_csv, index=False)
        return





    # get the significant bacteria full names
    features = [s[0] for s in significant_bacteria_and_rhos]
    # short_feature_names = [f.split(";")[-1] if len(f.split(";")[-1]) > 4 else f.split(";")[-2] for f in features]

    # extract the last meaningful name - long multi level names to the lowest level definition
    short_feature_names = []
    for f in features:
        i = 1
        if type(f) == list:
            print("l")
        while len(f.split(";")[-i]) < 5:  # meaningless name
            i += 1
        short_feature_names.append(f.split(";")[-i])

    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(significant_bacteria_and_rhos))
    c = [s[1] for s in significant_bacteria_and_rhos]
    coeff_color = []
    for x in c:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    # coeff_color = ['blue' for x in data >= 0]
    ax.barh(y_pos, c, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_feature_names)
    plt.yticks(fontsize=10)
    plt.title(algorithm + "\n" + task_name.capitalize() + " \nAverage coefficients for " + str(num_of_iters) + " runs")
    # ax.set_ylabel(ylabel)
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    # plt.show()
    if folder:
        plt.savefig(os.path.join(folder, algorithm + "_" + "bacteria_pos_neg_correlation_at_ " + task_name.replace(" ", "_") + "_avarage_of_" +
                                 str(num_of_iters) + "_runs.svg"), bbox_inches='tight', format='svg')
    else:
        plt.savefig(algorithm + "_" + "bacteria_pos_neg_correlation_at_ " + task_name.replace(" ", "_") + "_avarage_of_" +
                                 str(num_of_iters) + "_runs.svg", bbox_inches='tight', format='svg')





def create_combined_coeff_plots_from_files(files_paths, names, task_name, algorithm, num_of_iters):
    combined_df = pd.DataFrame(columns=["bacteria", "rhos", "name"])
    for path, name in zip(files_paths, names):
        df = pd.read_csv(path)
        for i in df.index:
            combined_df.loc[len(combined_df)] = [df.loc[i, "bacteria"], df.loc[i, "rhos"], name]
    print(combined_df)

    name_to_tag_map = {name: i for i, name in enumerate(names)}

    short_feature_names, bacterias = shorten_bact_names(combined_df["bacteria"])
    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(combined_df))
    c = combined_df["rhos"]
    coeff_color = combined_df["name"]
    coeff_color = [name_to_tag_map[c] for c in coeff_color]
    colors = ['aqua', 'darkviolet', 'gold', 'darkorange', 'red', 'greenyellow', 'darkgrey', 'darkgreen']
    coeff_color = [colors[c] for c in coeff_color]
    colors_dict = {name: colors[i] for i, name in enumerate(names)}

    ax.barh(y_pos, c, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_feature_names)
    plt.yticks(fontsize=10)
    plt.title(algorithm + "\n" + task_name.capitalize() + " \nAverage coefficients for " + str(num_of_iters) + " runs")
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)

    patches = []
    for key in reversed(list(colors_dict.keys())):
        patches.append(mpatches.Patch(color=colors_dict[key], label=key))
    ax.legend(handles=patches, fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.show()
    plt.savefig(algorithm + "_" + "bacteria_pos_neg_correlation_at_" +
                             task_name.replace(" ", "_") + "_avarage_of_" +
                             str(num_of_iters) + "_runs_all_types.svg", bbox_inches='tight', format='svg')

if __name__ == "__main__":
    os.chdir("..")
    folder = "allergy/allergy_type_before_treatment_task/SVM/k=linear_c=0.1_g=auto"
    os.chdir(folder)
    paths = ["SVM_allergy type before treatment task - Milk class_significant_bacteria_coeff_average_of_5_runs.csv",
             "SVM_allergy type before treatment task - Peanut class_significant_bacteria_coeff_average_of_5_runs.csv",
             "SVM_allergy type before treatment task - Sesame class_significant_bacteria_coeff_average_of_5_runs.csv",
             "SVM_allergy type before treatment task - Tree nut class_significant_bacteria_coeff_average_of_5_runs.csv"]
    names = ["Milk", "Peanut", "Sesame", "Tree nut"]
    create_combined_coeff_plots_from_files(paths, names, task_name="allergy type before treatment task",
                                           algorithm="SVM", num_of_iters=5)
