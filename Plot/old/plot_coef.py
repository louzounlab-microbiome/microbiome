from os.path import join
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches


def pop_idx(idx, objects_to_remove_idx_from):
    idx.reverse()
    for obj in objects_to_remove_idx_from:
        for i in idx:
            obj.pop(i)
    return objects_to_remove_idx_from


def shorten_bact_names(bacterias):
    # extract the last meaningful name - long multi level names to the lowest level definition
    short_bacterias_names = []
    for f in bacterias:
        i = 1
        while len(f.split(";")[-i]) < 3 or f.split(";")[-i] in ['Unassigned', 'NA']:  # meaningless name
            i += 1
            if i > len(f.split(";")):
                i -= 1
                break
        short_bacterias_names.append(f.split(";")[-i].strip(" "))
    # remove "k_bacteria" and "Unassigned" samples - irrelevant
    k_bact_idx = []
    for i, bact in enumerate(short_bacterias_names):
        if bact == 'k__Bacteria' or bact == 'Unassigned':
            k_bact_idx.append(i)

    if k_bact_idx:
        [short_bacterias_names, bacterias] = pop_idx(k_bact_idx, [short_bacterias_names, bacterias])

    return short_bacterias_names, bacterias


def create_coeff_plots_by_alogorithm(averages, bacterias, task_name, algorithm, num_of_iters, edge_percent, folder=False):
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
        result_csv = join(folder, algorithm + "_" + task_name.replace(" ", "_") + "_significant_bacteria_coeff_average_of_" +
                               str(num_of_iters) + "_runs.csv")
    else:
        result_csv = algorithm + "_" + task_name.replace(" ", "_") + "_significant_bacteria_coeff_average_of_" + str(num_of_iters) + "_runs.csv"

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
    significant_bacteria = [s[0] for s in significant_bacteria_and_rhos]

    # extract the last meaningful name - long multi level names to the lowest level definition
    short_feature_names, bacterias = shorten_bact_names(significant_bacteria)

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
    ax.barh(y_pos, c, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_feature_names)
    plt.yticks(fontsize=10)
    plt.title(algorithm + "\n" + task_name.capitalize() + " \nAverage coefficients for " + str(num_of_iters) + " runs")
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    # plt.show()
    if folder:
        plt.savefig(os.path.join(folder, algorithm + "_" + "bacteria_pos_neg_correlation_at_ " + task_name.replace(" ", "_") + "_avarage_of_" +
                                 str(num_of_iters) + "_runs.svg"), bbox_inches='tight', format='svg')
    else:
        plt.savefig(algorithm + "_" + "bacteria_pos_neg_correlation_at_ " + task_name.replace(" ", "_") + "_avarage_of_" +
                                 str(num_of_iters) + "_runs.svg", bbox_inches='tight', format='svg')
    plt.close()


def create_combined_coeff_plots_from_files(files_paths, names, task_name, algorithm, num_of_iters):
    # combine the results from each sub model into 1 big plot
    # each sub model gets it color(negative and positive correlations)
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
    coeff_color = combined_df["name"]  # color is decided by group
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
    plt.close()


def make_class_coef_plots_from_multiclass_model_binary_sub_models(files_paths, pair_names, names, folder, task_name,
                                           algorithm, num_of_iters=5):
    # combine the results from each sub model into 1 big data frame
    # the name_pair and the order of the files_paths fits each other, files-> ['bacteria', 'rhos'] columns
    # names order is irrelevant, should contain the names of each group
    combined_df = pd.DataFrame(columns=["bacteria", "rhos", "belong_to_group", "against_group"])
    for path, pair in zip(files_paths, pair_names):
        df = pd.read_csv(os.path.join(folder, path))
        for i in df.index:
            rho = df.loc[i, "rhos"]
            combined_df.loc[len(combined_df)] = [df.loc[i, "bacteria"], rho, pair[0], pair[1]]
            combined_df.loc[len(combined_df)] = [df.loc[i, "bacteria"], -rho, pair[1], pair[0]]

    name_to_tag_map = {name: i for i, name in enumerate(names)}

    # create plot for each group, subset the df for relevant rows
    for group in names:
        group_df = combined_df[combined_df["belong_to_group"] == group]

        short_feature_names, bacterias = shorten_bact_names(group_df["bacteria"])
        group_df["bacteria"] = short_feature_names

        # in order to merge values of identical bacterias, create another df with hold the rho value
        # for each unique_bacteria for each group
        unique_bacteria = np.unique(short_feature_names)
        all_rhos_of_all_bact_df = pd.DataFrame(columns=names, index=unique_bacteria)
        for bact in unique_bacteria:
            # get all rows in df with this bacteria
            bact_rows = group_df[group_df['bacteria'] == bact]

            """  # messed up the mutual bacteria.. figure out why?
            # if we have different rho values when the other attributes are the same,
            #  enter the average rho as one entrance
            
            bact_rows_grouped_by = bact_rows.groupby(["bacteria", "belong_to_group", "against_group"]).mean()
            for multi_index in list(bact_rows_grouped_by.index):
                bact_rows = pd.DataFrame(columns=["bacteria", "rhos", "belong_to_group", "against_group"])
                bact_rows.loc[len(bact_rows)] = [multi_index[0], float(bact_rows_grouped_by.loc[multi_index]), multi_index[1], multi_index[2]]
            """

            # create rhos list for all of the groups in 'names'
            rho_list = []
            for name in names:
                found = False
                for row in bact_rows.itertuples():
                    if name == getattr(row, 'against_group'):  # has a significant rho value
                        rho_list.append(getattr(row, 'rhos'))
                        found = True
                if not found:  # doesn't has a significant rho value, add zero-padding
                    rho_list.append(0)

            all_rhos_of_all_bact_df.loc[bact] = rho_list  # add rhos to the bacteria row

        # calculate accumulating sum in order to plot the rhos with out them covering each other
        for idx in all_rhos_of_all_bact_df.index:
            all_rhos_of_all_bact_df.loc[idx] = np.cumsum(all_rhos_of_all_bact_df.loc[idx])

        # because of the accumulating, sort from last column to first
        sort_column_order = list(all_rhos_of_all_bact_df.columns)
        sort_column_order.reverse()
        all_rhos_of_all_bact_df = all_rhos_of_all_bact_df.sort_values(by=sort_column_order, ascending=True)
        all_rhos_of_all_bact_df.to_csv(group + "_accumulating_rhos_table.csv")

        # plot
        final_bact = list(all_rhos_of_all_bact_df.index)
        left_padding = 0.4
        fig, ax = plt.subplots()
        y_pos = np.arange(len(final_bact))
        colors = ['blue', 'gold', 'red', 'green', 'magenta', 'darkgreen']
        colors_dict = {name: colors[i] for i, name in enumerate(names)}
        # plot reverse because of the accumulating from start to end, this way the long lines won't cover the short ones
        name_list, rhos_list = [], []
        for name, rhos in zip(names, all_rhos_of_all_bact_df.iteritems()):
            name_list.append(name)
            rhos_list.append(rhos)
        name_list.reverse()
        rhos_list.reverse()

        for name, rhos in zip(name_list, rhos_list):
            if name != group:
                    coeff_color = [colors[name_to_tag_map[name]]] * len(all_rhos_of_all_bact_df)
                    ax.barh(y_pos, list(rhos.__getitem__(1)), color=coeff_color)

        # add info to the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(final_bact)
        plt.yticks(fontsize=10)
        plt.title(algorithm + "\n" + task_name.capitalize() + " - " + group + " class\nStacked Average coefficients for " + str(
            num_of_iters) + " runs")
        ax.set_xlabel("Coeff value")
        fig.subplots_adjust(left=left_padding)

        # add dictionary from name of group to color on the side
        patches = []
        for key in reversed(list(colors_dict.keys())):
            if key == group:
                continue
            patches.append(mpatches.Patch(color=colors_dict[key], label=key))
        ax.legend(handles=patches, fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))

        # plt.show()
        plt.savefig(os.path.join(folder, algorithm + "_" + "bacteria_pos_neg_correlation_at_" +
                    task_name.replace(" ", "_") + "_" + group + "_class_stacked_avarage_of_" +
                    str(num_of_iters) + "_runs_all_types.svg"), bbox_inches='tight', format='svg')
        plt.close()


if __name__ == "__main__":
    os.chdir("..")
    folder = "allergy/allergy_type_before_treatment_task/SVM/chosen_result"
    os.listdir(folder)
    os.chdir(folder)
    paths = ['SVM_allergy type before treatment task-Tree nut class vs. Sesame class_significant_bacteria_coeff_average_of_5_runs.csv',
 'SVM_allergy type before treatment task-Tree nut class vs. Peanut class_significant_bacteria_coeff_average_of_5_runs.csv',
 'SVM_allergy type before treatment task-Milk class vs. Tree nut class_significant_bacteria_coeff_average_of_5_runs.csv',
 'SVM_allergy type before treatment task-Milk class vs. Sesame class_significant_bacteria_coeff_average_of_5_runs.csv',
 'SVM_allergy type before treatment task-Milk class vs. Peanut class_significant_bacteria_coeff_average_of_5_runs.csv',
 'SVM_allergy type before treatment task-Peanut class vs. Sesame class_significant_bacteria_coeff_average_of_5_runs.csv']
    names = ["Milk", "Peanut", "Sesame", "Tree nut"]
    pair_names = [["Tree nut", "Sesame"], ["Tree nut", "Peanut"], ["Milk", "Tree nut"], ["Milk", "Sesame"], ["Milk", "Peanut"], ["Peanut", "Sesame"]]
    make_class_coef_plots_from_multiclass_model_binary_sub_models(paths, pair_names, names, folder, task_name="allergy type before treatment task",
                                           algorithm="SVM", num_of_iters=5)
    """
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
    """
