import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.intervention_nn import run_NN
import nni
from LearningMethods.intervention_rnn import run_RNN
from Microbiome_Intervention.Create_learning_data_from_data_set import get_adapted_X_y_for_wanted_learning_task


def get_10_representing_bacteria_indexes(df):
    representing_bacteria_indexes = []
    df = df.sort_values(by=["NN_MSE"])
    bacteria = df["BACTERIA"]
    bact_num = len(df)
    group_size = int(bact_num / 10)
    for i in range(1, 11):
        representing_bacteria_indexes.append(bacteria[group_size*i - 4])  # first run -3, second run -2, third run -4
    return representing_bacteria_indexes


def run_single_bacteria(tax, bacteria_sorted_by_mse, best_bacteria_path, X_y_files_list_path, NN_or_RNN, results_df_title):

    with open(os.path.join(tax, X_y_files_list_path), "r") as file:
        multi_path = file.readline()
        multi_path = multi_path.strip('\n')

    with open(os.path.join(tax, "bacteria.txt"), "r") as b_file:
        bacteria = b_file.readlines()
        bacteria = [b.rstrip() for b in bacteria]

    # ------------------------------------ decide on mission ------------------------------------
    nni_ = False  # check model or run nni for real
    GPU = True if nni_ else False
    report_loss = True
    report_correlation = not report_loss
    k_fold = False  # run k fold
    RNN = True if NN_or_RNN == "RNN" else False
    NN = True if NN_or_RNN == "NN" else False
    representing_bacteria = False

    if nni_:
        params = nni.get_next_parameter()
    else:
        params = {"NN_STRUCTURE": "002L050H050H",
                  "TRAIN_TEST_SPLIT": 0.7,
                  "EPOCHS": 200,
                  "LEARNING_RATE": 1e-3,
                  "OPTIMIZER": "Adam",
                  "REGULARIZATION": 0.05,
                  "DROPOUT": 0}

    # ------------------------------------ data loading ------------------------------------
    # run a prediction of a single bacteria at a time
    # consider the average loss and correlation of all runs as the performance measurement

    if representing_bacteria:
        title = os.path.join(tax, ("NN_" if NN else "RNN_") + "10_representing_bacteria_" + results_df_title + ".csv")
    else:
        title = os.path.join(tax, ("NN_" if NN else "RNN_") + results_df_title + ".csv")

    all_results_df = pd.DataFrame(
        columns=["BACTERIA", "STRUCTURE", "LEARNING_RATE", "REGULARIZATION", "DROPOUT",
                 "TEST_MSE", "TEST_CORR", "TEST_R2",
                 "TRAIN_MSE", "TRAIN_CORR", "TRAIN_R2"])

    """
    previous_results = pd.read_csv(os.path.join(tax, bacteria_sorted_by_mse))
    sorted_df = pd.read_csv((os.path.join(tax, best_bacteria_path)))
    if representing_bacteria:
        indexes = get_10_representing_bacteria_indexes(previous_results)
    else:
    """
    indexes = range(len(bacteria))

    for STRUCTURE in ["001L025H", "001L050H", "001L100H", "001L200H", "002L025H025H", "002L050H050H", "002L100H100H"]:
        for LEARNING_RATE in [1e-2]:
            for REGULARIZATION in [0, 0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 2.5]:
                for DROPOUT in [0, 0.001, 0.01, 0.1, 0.2]:
                    epochs = 70
                    params = {"STRUCTURE": STRUCTURE,
                              "TRAIN_TEST_SPLIT": 0.7,
                              "EPOCHS": epochs,
                              "LEARNING_RATE": LEARNING_RATE,
                              "OPTIMIZER": "Adam",
                              "REGULARIZATION": REGULARIZATION,
                              "DROPOUT": DROPOUT}
                    print(params)

                    df = pd.DataFrame(
                        columns=["BACTERIA", "STRUCTURE", "LEARNING_RATE", "REGULARIZATION", "DROPOUT",
                                 "TEST_MSE", "TEST_CORR", "TEST_R2",
                                 "TRAIN_MSE", "TRAIN_CORR", "TRAIN_R2"])

                    for i in indexes:  # for loop every single bacteria
                        print(str(i) + " / " + str(len(indexes)))
                        #best_bacteria_num = sorted_df["BACTERIA_NUMBER"][i]
                        #reg_mse = sorted_df["MSE"][i]
                        #reg_rho = sorted_df["RHO"][i]
                        path = "time_serie_X_y_for_bacteria_number_" + str(i) + ".csv"
                        X, y, missing_values, name = get_adapted_X_y_for_wanted_learning_task(tax, path, "time_serie")
                        NUMBER_OF_SAMPLES = X.shape[0]
                        NUMBER_OF_TIME_POINTS = X.shape[1]
                        NUMBER_OF_BACTERIA = X.shape[2]

                        # ------------------------------------ send to network ------------------------------------
                        if RNN:
                            res_map = run_RNN(X, y, missing_values, name, tax, params,
                                                         NUMBER_OF_SAMPLES, NUMBER_OF_TIME_POINTS, NUMBER_OF_BACTERIA,
                                    GPU_flag=GPU, task_id=str(i))
                            rnn_loss = res_map["TEST"]["loss"]
                            rnn_corr = res_map["TEST"]["corr"]
                            rnn_r2 = res_map["TEST"]["r2"]

                            t_rnn_loss = res_map["TRAIN"]["loss"]
                            t_rnn_corr = res_map["TRAIN"]["corr"]
                            t_rnn_r2 = res_map["TRAIN"]["r2"]

                            # print(params)
                            print("loss=" + str(nn_loss))
                            print("corr=" + str(nn_corr))
                            print("r2=" + str(nn_r2))

                            df.loc[len(df)] = [int(i),
                                               STRUCTURE, LEARNING_RATE, REGULARIZATION, DROPOUT,
                                               rnn_loss, rnn_corr, rnn_r2,
                                               t_rnn_loss, t_rnn_corr, t_rnn_r2]

                        if NN:
                            flat_time_points_values_num = NUMBER_OF_SAMPLES * NUMBER_OF_TIME_POINTS

                            X = X.reshape(flat_time_points_values_num, NUMBER_OF_BACTERIA)
                            y = y.reshape(flat_time_points_values_num)
                            missing_values = missing_values.reshape(flat_time_points_values_num)

                            person_indexes = np.linspace(0, flat_time_points_values_num - 1, flat_time_points_values_num).\
                                reshape(NUMBER_OF_SAMPLES, NUMBER_OF_TIME_POINTS).astype(int).tolist()

                            res_map = run_NN(X, y, missing_values, params, name, tax,
                                                      NUMBER_OF_SAMPLES, NUMBER_OF_TIME_POINTS, NUMBER_OF_BACTERIA,
                                                      GPU_flag=GPU, k_fold=k_fold, task_id=str(i), person_indexes=person_indexes)
                            nn_loss = res_map["TEST"]["loss"]
                            nn_corr = res_map["TEST"]["corr"]
                            nn_r2 = res_map["TEST"]["r2"]

                            t_nn_loss = res_map["TRAIN"]["loss"]
                            t_nn_corr = res_map["TRAIN"]["corr"]
                            t_nn_r2 = res_map["TRAIN"]["r2"]

                            print(int(i))
                            df.loc[len(df)] = [int(i),
                                               STRUCTURE, LEARNING_RATE, REGULARIZATION, DROPOUT,
                                               nn_loss, nn_corr, nn_r2,
                                               t_nn_loss, t_nn_corr, t_nn_r2]

                    if nni_:
                        if report_loss:
                            nni.report_final_result(df["TEST_MSE"].mean())
                        elif report_correlation:
                            nni.report_final_result(df["TEST_CORR"].mean())

                    all_results_df.loc[len(all_results_df)] = ["average", STRUCTURE, LEARNING_RATE, REGULARIZATION, DROPOUT,
                                                   df["TEST_MSE"].mean(), df["TEST_CORR"].mean(), df["TEST_R2"].mean(),
                                                   df["TRAIN_MSE"].mean(), df["TRAIN_CORR"].mean(), df["TRAIN_R2"].mean()]
                    all_results_df.to_csv(title, index=False)

    print(not all_results_df.empty)
    if len(all_results_df) > 0:
        all_results_df.to_csv(os.path.join(tax, "NNI", title), index=False)


if __name__ == "__main__":  # create an option for nni - get params from file
    tax = os.path.join('GDM_data', 'tax=5')
    bacteria_sorted_by_mse = "bacteria_sorted_by_mse.csv"
    best_bacteria_path = "run_all_types_of_regression_10_fold_test_size_0.5_bacteria_conclusions.csv"
    X_y_files_list_path = 'multi_bacteria_time_serie_files_names.txt'
    run_single_bacteria(tax, bacteria_sorted_by_mse, best_bacteria_path, X_y_files_list_path,
                        NN_or_RNN="NN", results_df_title="bacteria_grid_search_df")

