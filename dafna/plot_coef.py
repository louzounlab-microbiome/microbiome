from os.path import join
import os
import numpy as np
import matplotlib.pyplot as plt


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
        with open(join(folder, algorithm + "_" + task_name + "_significant_bacteria_coeff_average_of_" +
                               str(num_of_iters) + "_runs.txt"), "w") as file:
            for s in significant_bacteria_and_rhos:
                file.write(str(s[1]) + "," + str(s[0]) + "\n")
    else:
        with open(algorithm + "_" + task_name + "_significant_bacteria_coeff_average_of_" + str(num_of_iters)
                  + "_runs.txt", "w") as file:
            for s in significant_bacteria_and_rhos:
                file.write(str(s[1]) + "," + str(s[0]) + "\n")


    # get the significant bacteria full names
    features = [s[0] for s in significant_bacteria_and_rhos]
    # short_feature_names = [f.split(";")[-1] if len(f.split(";")[-1]) > 4 else f.split(";")[-2] for f in features]

    # extract the last meaningful name - long multi level names to the lowest level definition
    short_feature_names = []
    for f in features:
        i = 1
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

