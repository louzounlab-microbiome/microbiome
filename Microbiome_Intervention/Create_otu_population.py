import copy
import pickle
import random
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from Microbiome_Intervention.Create_learning_data_from_data_set import get_adapted_X_y_for_wanted_learning_task
from Microbiome_Intervention.Simple_prediction_of_natural_dynamics import create_data_frames, \
    predict_interaction_network_structure_using_coeffs, predict_interaction_network_structure_using_change_in_data


def generate_otus(number_of_bacteria, number_of_interactions, number_of_samples, number_of_time_points, folder="Simulations"):
    """
    The bacteria counts is created by generating a random samples for the starting time point (values between ~0-1000)
    then, creating 3 influences on the change in time:
    growth rate for each bacteria, a random rate between 0-1.
    reduction rate - shared between all bacteria ~0.2.
    interactions rate - a matrix of size n*n where n is the number of bacteria.
    For each cell in index [i,j], the cell content reflect the influence j has on i. the content options are:
    0, no influence , 1=increase in count, -1=decrease in count.
    the number of influencers on each bacteria is selected in advanced ~3.

    I used a mathematical model that given the data and some function f calculate  f(t) which is the population at the
    time t.
    The function calculates the derivative and thus solves the equations.
    dy/dt = y * (growth rates - reduction rate * y +(np.sum(interactions rates * y, axis=1)))
    Using this equation, we will generate the population for the number of time points requested, and generate all the
    types of files required for the different learning tasks.
    :param number_of_bacteria: (int) wanted number of bacteria.
    :param number_of_interactions: (int) wanted number of interactions.
    :param number_of_samples: (int) wanted number of samples.
    :param number_of_time_points: (int) wanted number of time_points.
    :param folder: (string) folder to save files in.
    :return: no object is returned.
    """
    growth_rates = np.array([random.random() for i in range(number_of_bacteria)])
    reduction_rate = 0.01
    interactions_rates = np.array([[0 for i in range(number_of_bacteria)] for i in range(number_of_bacteria)])
    # put 1 in the places we what to create an interaction
    g = pd.DataFrame(columns=["FROM", "TO", "RELATION"])

    edges = []
    for i, interactions in enumerate(interactions_rates):
        for n in range(number_of_interactions):
            idx = int(random.random() * number_of_bacteria)
            influence_type = random.sample([1, -1], 1)
            interactions[idx] = influence_type[0]
            edges.append((idx, i, influence_type))  # idx affects i
            g.loc[len(g)] = [idx, i, influence_type[0]]

    pickle.dump(edges, open(os.path.join(folder, "simulation_edegs.pkl"), "wb"))
    g.to_csv(os.path.join(folder, "simulation_graph.csv"), index=False)
    pd.DataFrame(data=interactions_rates, columns=range(number_of_bacteria)).\
        to_csv(os.path.join(folder, "simulation_interactions.csv"), index=False)

    def model(t, y):
        scalar = 1 / number_of_bacteria
        dydt = y * (growth_rates - reduction_rate * y + scalar * (np.sum(interactions_rates * y, axis=1)))
        return dydt

    # create population in time 0
    population = [0, 1, 5, 10, 100]
    weights = [0.80, 0.15, 0.02, 0.02, 0.01]

    otu_t_0 = np.array([[sum([random.choices(population, weights)[0] for r in range(100)])
                         for i in range(number_of_bacteria)] for j in range(number_of_samples)])

    otu_time_serie_columns = ["t=" + str(i) for i in range(number_of_time_points)]
    otu_raw_data = pd.DataFrame(columns=["sample", "t", "value"])
    t = np.linspace(0, 1.3, number_of_time_points)
    for i, sample in enumerate(otu_t_0):
        # y = odeint(model, sample, t, tfirst=True).T
        result = solve_ivp(model, t_span=(0, 0.8), y0=sample, t_eval=np.linspace(0, 0.8, number_of_time_points))
        y = result["y"].T
        """
        plt.style.use('seaborn-darkgrid')
        palette = plt.get_cmap('tab20b')
        num = 0
        for bact in y:
            num += 1
            plt.plot(list(range(len(bact))), bact, marker='', color=palette(num % 20), linewidth=1, alpha=0.5)
            # plt.show()

        plt.legend(loc=2, ncol=2)
        #plt.plot(t, y)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.title("Sample " + str(i + 1))
        plt.show()
        """
        for j in range(number_of_time_points):
            otu_raw_data.loc[len(otu_raw_data)] = [i, j, np.array(y[j])]

    print("simulation_time_serie_X_y_for_all_bacteria")
    otu_time_serie = pd.DataFrame(columns=otu_time_serie_columns)

    data = []
    for sample in np.array(otu_raw_data["value"]):
        data.append(sample)
    preproccesed_data = preprocessing.scale(pd.DataFrame(data=data), axis=0)
    for i in range(0, number_of_samples*number_of_time_points, number_of_time_points):
        values = []
        for j in range(i, i + number_of_time_points):
            values.append(preproccesed_data[j])
        otu_time_serie.loc[len(otu_time_serie)] = values

    otu_delta_time_serie = pd.DataFrame(index=otu_time_serie.index, columns=otu_time_serie_columns[:-1])
    for col_i in range(len(otu_time_serie.columns) - 1):
        # create the change in time
        t_2 = np.array(otu_time_serie[otu_time_serie_columns[col_i + 1]])
        t_1 = np.array(otu_time_serie[otu_time_serie_columns[col_i]])
        otu_delta_time_serie[otu_time_serie_columns[col_i]] = list(t_2 - t_1)

    df = pd.DataFrame(columns=['X', 'y'])
    for sample_i in otu_time_serie.index:
        X = list(otu_time_serie.drop(columns=[otu_time_serie_columns[-1]]).loc[sample_i].values.tolist())
        Y = list(otu_delta_time_serie.loc[sample_i].values.tolist())
        df.loc[(len(df))] = [';'.join(map(str, X)).replace("\n", "").replace("   ", " ").replace("  ", " "), ';'.join(map(str, Y)).replace("\n", "").replace(",", "")]
    file_name = "simulation_time_serie_X_y_for_all_bacteria.csv"
    df.to_csv(os.path.join(folder, file_name), index=False)

    print("X_y_for_bacteria")
    df_paths_list = []
    for bact_num, bact in enumerate(range(number_of_bacteria)):
        df = pd.DataFrame(columns=['Time Point', 'ID', 'X', 'y'])
        for time_point in otu_time_serie_columns[:-1]:
            time_point_X = otu_time_serie[time_point].values
            X = copy.deepcopy(time_point_X)
            # get delta change form bacteria-bact num in all samples
            pre_y = otu_delta_time_serie[time_point].values
            Y = []
            for row in pre_y:
                Y.append([row[bact_num]])

            # save to csv file for later
            for x, y in zip(X, Y):
                df.loc[(len(df))] = [time_point, bact_num,
                                     x.__str__()[2:-1].replace("\n", "").replace("  ", " "), y.__str__()[1:-1]]

        file_name = "X_y_for_bacteria_number_" + str(bact_num) + ".csv"
        df.to_csv(os.path.join(folder, file_name), index=False)
        df_paths_list.append(file_name)

    with open(os.path.join(folder, "files_names.txt"), "w") as paths_file:
        for path in df_paths_list:
            paths_file.write(path + '\n')

    print("time_serie_X_y_for_bacteria")
    df_paths_list = []
    for bact_num, bact in enumerate(range(number_of_bacteria)):
        df = pd.DataFrame(columns=['X', 'y'])
        for sample_i in otu_time_serie.index:
            X = list(otu_time_serie.drop(columns=[otu_time_serie_columns[-1]]).loc[sample_i].values.tolist())
            Y = [row[bact_num] for row in otu_delta_time_serie.loc[sample_i].values.tolist()]
            df.loc[(len(df))] = [';'.join(map(str, X)).replace("\n", "").replace("   ", " ").replace("  ", " "),
                                 ';'.join(map(str, Y)).replace("\n", "").replace(",", "")]
        file_name = "time_serie_X_y_for_bacteria_number_" + str(bact_num) + ".csv"
        df.to_csv(os.path.join(folder, file_name), index=False)
        df_paths_list.append(file_name)
    with open(os.path.join(folder, "time_serie_files_names.txt"), "w") as paths_file:
        for path in df_paths_list:
            paths_file.write(path + '\n')


def run_regression_coef_net(reg_type, k_fold, test_size, folder, data_set_name):
    """
    :param k_fold: (int) returns K fold of the split to train and test
    :param test_size: ((float) the size of the test for the split to train and test
    :param folder: (string) files folder
    :param data_set_name: (string) dataset name - same as in folders.
    :return: no object is returned, AUC of the net is saved to "all_Networks_AUC.csv" file in the main folder
    """
    task = "interaction_network_structure_coef"
    all_times_all_bact_results_path = os.path.join(folder,
                                                   reg_type.replace(" ", "_") + "_" + task + "_" + str(k_fold) +
                                                   "_fold_test_size_" + str(test_size)
                                                   + "_results_df.csv")
    important_bacteria_reults_path = os.path.join(folder, reg_type.replace(" ", "_") + "_" + task + "_" + str(k_fold) +
                                                  "_fold_test_size_" + str(test_size)
                                                  + "_significant_bacteria_prediction_results_df.csv")

    bacteria = [i for i in range(100)]

    create_data_frames(all_res_path=all_times_all_bact_results_path,
                       important_bacteria_reults_path=important_bacteria_reults_path)

    paths = ["X_y_for_bacteria_number_" + str(i) + ".csv" for i in range(100)]

    train_binary_significant_from_all_bacteria = []
    test_b_list_from_all_bacteria = []

    for i, [bact, path] in enumerate(zip(bacteria, paths)):
        print(str(i) + " / " + str(len(bacteria)))

        all_times_all_bacteria_all_models_results_df = pd.read_csv(all_times_all_bact_results_path)
        important_bacteria_reults_df = pd.read_csv(important_bacteria_reults_path)
        X_trains, X_tests, y_trains, y_tests, name = \
            get_adapted_X_y_for_wanted_learning_task(folder, path, "regular", k_fold, test_size)

        results_df, train_binary_significant_list, test_b_list = \
            predict_interaction_network_structure_using_coeffs(X_trains, X_tests, y_trains, y_tests, i,
                                                               all_times_all_bacteria_all_models_results_df,
                                                               all_times_all_bact_results_path,
                                                               important_bacteria_reults_df,
                                                               important_bacteria_reults_path, bact, bacteria,
                                                               reg_type)
        # save bacteria y true nd y pred
        train_binary_significant_from_all_bacteria.append(list(np.array(train_binary_significant_list).flat))
        test_b_list_from_all_bacteria.append(list(np.array(test_b_list).flat))

    train_binary_significant_from_all_bacteria = list(np.array(train_binary_significant_from_all_bacteria).flat)
    test_b_list_from_all_bacteria = list(np.array(test_b_list_from_all_bacteria).flat)
    total_auc = roc_auc_score(y_true=train_binary_significant_from_all_bacteria,
                              y_score=test_b_list_from_all_bacteria)

    Networks_AUC_df = pd.read_csv("all_Networks_AUC.csv")
    Networks_AUC_df.loc[len(Networks_AUC_df)] = ["coefficients", reg_type, data_set_name, test_size, k_fold, total_auc,
                                                 datetime.utcnow().strftime("%d/%m/%Y %H:%M:%S")]
    Networks_AUC_df.to_csv("all_Networks_AUC.csv", index=False)


if __name__ == "__main__":
    folder = os.path.join("Simulations", "4")
    generate_otus(100, 3, 200, 4,  folder=folder)
    algorithms_list = ["linear regression", "ridge regression", "ard regression",
                       "lasso regression", "bayesian ridge regression",
                       "svr regression"]  # "decision tree regression", "random forest regression"
    for reg_type in algorithms_list:
        run_regression_coef_net(reg_type, k_fold=20, test_size=0.3, folder=folder, data_set_name="Simulations 1")
    predict_interaction_network_structure_using_change_in_data(list(range(100)), os.path.join("Simulations", "1"),
                                                               CHANGE=0.5)

