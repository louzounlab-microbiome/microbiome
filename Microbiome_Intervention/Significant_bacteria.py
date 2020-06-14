import os
import pickle
from random import shuffle
import pandas as pd
import numpy as np

from LearningMethods import shorten_single_bact_name
from Plot.plot_bacteria_intraction_network import plot_bacteria_intraction_network


def check_if_bacteria_correlation_is_significant(df_path, model):
    """
    check if bacteria is 2 std step up or down from the mean the rho correlations.
    :param df_path: (string) path to coefficients data frame.
    :param model: (string) model name.
    :return: list of the significant bacteria.
    """
    # see if the real rho value is 2 std steps from the mixed rho mean, if so -> rho value is significant
    significant_bacteria = []
    df = pd.read_csv(df_path)
    sub_df = df[df["ALGORITHM"] == model]
    bacteria = sub_df["BACTERIA"]
    rhos = sub_df["RHO"]
    mixed_rho_mean = sub_df["RANDOM_RHO"].mean()
    mixed_rho_std = sub_df["RANDOM_RHO"].std()
    for bact, r in zip(bacteria, rhos):
        if r > mixed_rho_mean + (2*mixed_rho_std) or r < mixed_rho_mean - (2*mixed_rho_std):
            significant_bacteria.append(bact)

    return significant_bacteria


def get_significant_beta_from_file(df_path, model, folder, bacteria_path):
    """
    After running 'interaction_network_structure_coef' task and collecting all coefficient-
    In order to build an interaction network, while predicting bacteria a, if the coefficient value for bacteria b
    were 2 std step up or down from the mean the interaction concluded existing.

    :param df_path: (string) path to coefficients data frame.
    :param model: (string) model name.
    :param folder: (string) main dataset folder "DATASET/tax=x"
    :param bacteria_path: (string) path to bacteria list txt file.
    :return:d Doesn't return an object, create 'important_bacteria_df' for the existing interaction. 
    'all_rhos_df' for the average coefficients.
    node_list, edge_list and color_list are created in order to visulaze the network.
    """
    important_bacteria_df = pd.DataFrame(columns=["MODEL BACTERIA", "SIGNIFICANT BACTERIA", 'ALGORITHM',
                                                  'BETA MEAN', 'BETA STD', 'BETA'])

    df = pd.read_csv(df_path)
    bacteria = list(set(df["BACTERIA"].values))
    all_rhos_df = pd.DataFrame(index=bacteria, columns=bacteria)
    edges = []
    color_map = {"-+ -> -+": '#FF4233',  # red
                 "-+ -> +-": '#3E89C0',  # blue
                 }

    y_true = np.array([[0 for i in range(len(bacteria))] for i in range(len(bacteria))])

    sub_df = df[df["ALGORITHM"] == model]
    for i, b in enumerate(bacteria):
        print(str(i) + " / " + str(len(bacteria)))
        bacteria_df = sub_df[sub_df["BACTERIA"] == b]
        beta_list = [beta.split(";") for beta in bacteria_df["BETA"]]
        for l_i, l in enumerate(beta_list):
            if l == [' ']:
                beta_list[l_i] = 'nan'
                continue
            for s_i, s in enumerate(l):
                l[s_i] = float(s)

        numeric_beta_list = [ls for ls in beta_list if len(ls) == len(bacteria)]

        bact_average_beta = np.mean(np.array((numeric_beta_list)), axis=0)
        all_rhos_df.loc[b] = bact_average_beta

        mean = np.mean(bact_average_beta)
        std = np.std(bact_average_beta)
        for b_i, sub_b in enumerate(bact_average_beta):  # iterate rhos
            if sub_b > mean + (2 * std) or sub_b < mean - (2 * std):
                important_bacteria_df.loc[len(important_bacteria_df)] =\
                    [b, bacteria[b_i], model, mean, std, sub_b]
                y_true[i][b_i] = 1
                if sub_b > 0:
                    edges.append((b_i, i, {'color': color_map["-+ -> -+"]}))
                else:
                    edges.append((b_i, i, {'color': color_map["-+ -> +-"]}))

    if not os.path.exists(folder):
            os.mkdir(folder)
    return_path = os.path.join(folder, "all_bacteria_rhos_" + model.replace(" ", "_") + ".csv")
    all_rhos_df.to_csv(return_path)
    important_bacteria_df.to_csv(os.path.join(folder, "important_bacteria_" + model.replace(" ", "_") + ".csv"))

    # plot heat map of all_rhos_df
    all_rhos_df.index = [shorten_single_bact_name(bact) for bact in all_rhos_df.index]
    all_rhos_df.columns = [shorten_single_bact_name(bact) for bact in all_rhos_df.columns]
    all_rhos_df = all_rhos_df.dropna()

    node_list = [i for i in range(len(bacteria))]
    edge_list = []
    color_list = []

    num_of_edges_list = [0 for i in range(len(bacteria))]

    for e_ in edges:
        if e_[0] != e_[1]:
            num_of_edges_list[e_[0]] += 1
            num_of_edges_list[e_[1]] += 1

    for e_ in edges:
        if (num_of_edges_list[e_[0]] > 0) or (num_of_edges_list[e_[1]] > 0):
            edge_list.append((e_[0], e_[1]))
            color_list.append((e_[2]["color"]))

    shuffle(edge_list)
    title = "significant_bacteria_" + model
    pickle.dump(node_list, open(os.path.join(folder, title + "_node_list.pkl"), "wb"))
    pickle.dump(edge_list, open(os.path.join(folder, title + "_edge_list.pkl"), "wb"))
    pickle.dump(color_list, open(os.path.join(folder, title + "_color_list.pkl"), "wb"))
    v = [100] * len(bacteria)

    with open(bacteria_path, "r") as b_file:
        bacteria = b_file.readlines()
        bacteria = [b.rstrip() for b in bacteria]

    plot_bacteria_intraction_network(bacteria, node_list, v, edge_list, color_list, title, folder)

