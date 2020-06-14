import os
import sys
import pandas as pd
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.intervention_nn import run_NN
import nni
from LearningMethods.intervention_rnn import run_RNN
from Microbiome_Intervention.Create_learning_data_from_data_set import get_adapted_X_y_for_wanted_learning_task


def run_multi_bacteria(tax, multi_time_serie_path, multi_reg_path, NN_or_RNN, results_df_title):
    # ------------------------------------ decide on mission ------------------------------------
    nni_ = False  # check model or run nni for real
    GPU = True if nni_ else False
    report_loss = True
    report_correlation = not report_loss
    k_fold = False  # run k fold
    RNN = True if NN_or_RNN == "RNN" else False
    NN = True if NN_or_RNN == "NN" else False


    if nni_:
        params = nni.get_next_parameter()
    else:
        params = {"STRUCTURE": "002L050H050H",
                  "TRAIN_TEST_SPLIT": 0.7,
                  "EPOCHS": 2,
                  "LEARNING_RATE": 1e-3,
                  "OPTIMIZER": "Adam",
                  "REGULARIZATION": 0.05,
                  "DROPOUT": 0}
    # ------------------------------------ data loading ------------------------------------
    # run a prediction of a single bacteria at a time
    # consider the average loss and correlation of all runs as the performance measurement


    # ------------------------------------ data loading ------------------------------------
    if RNN:
        X, y, missing_values, name = get_adapted_X_y_for_wanted_learning_task(tax, multi_time_serie_path, "multi_bact_time_serie")
        NUMBER_OF_SAMPLES = X.shape[0]
        NUMBER_OF_TIME_POINTS = X.shape[1]
        NUMBER_OF_BACTERIA = X.shape[2]


        # ------------------------------------ missing values adjustment ------------------------------------
        missing_values = missing_values.tolist()
        for s_i in range(NUMBER_OF_SAMPLES):
            for t_i in range(NUMBER_OF_TIME_POINTS):
                val = missing_values[s_i][t_i]
                missing_values[s_i][t_i] = [val for i in range(NUMBER_OF_BACTERIA)]

        missing_values = np.array(missing_values)

    if NN:
        X, y, person_indexes, name = \
            get_adapted_X_y_for_wanted_learning_task(tax, multi_reg_path, "multi_bact_regular")
        NUMBER_OF_SAMPLES = X.shape[0]
        NUMBER_OF_BACTERIA = X.shape[1]
        missing_values = np.array([[1 for i in range(NUMBER_OF_BACTERIA)] for j in range(NUMBER_OF_SAMPLES)])

    # ------------------------------------ send to network ------------------------------------
    title = os.path.join(tax, ("NN_" if NN else "RNN_") + results_df_title + ".csv")

    df = pd.DataFrame(columns=["TASK", "STRUCTURE", "REGULARIZATION", "DROPOUT", "EPOCHS",
                               "TEST_RMSE", "TEST_RHO", "TEST_R2",
                               "TRAIN_RMSE", "TRAIN_RHO", "TRAIN_R2"])
    df.to_csv(title, index=False)

    for STRUCTURE in ["001L025H", "001L050H", "001L100H", "001L200H", "002L025H025H", "002L050H050H", "002L100H100H"]:
        for LEARNING_RATE in [1e-2]:
            for REGULARIZATION in [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 2.5]:
                for DROPOUT in [0, 0.001, 0.01, 0.1, 0.2]:
                    epochs = 70
                    params = {"STRUCTURE": STRUCTURE,
                              "TRAIN_TEST_SPLIT": 0.7,
                              "EPOCHS": epochs,
                              "LEARNING_RATE": LEARNING_RATE,
                              "OPTIMIZER": "Adam",
                              "REGULARIZATION": REGULARIZATION,
                              "DROPOUT": DROPOUT}

                    if RNN:
                        folder = os.path.join(tax, "RNN")
                        res_map = run_RNN(X, y, missing_values, name, folder, params,
                                                     NUMBER_OF_SAMPLES, NUMBER_OF_TIME_POINTS,
                                                     NUMBER_OF_BACTERIA,
                                                     GPU_flag=GPU, task_id="all_bact")
                        rnn_loss = res_map["TEST"]["loss"]
                        rnn_corr = res_map["TEST"]["corr"]

                        t_rnn_loss = res_map["TRAIN"]["loss"]
                        t_rnn_corr = res_map["TRAIN"]["corr"]

                        df.loc[len(df)] = ["Multi Bacteria - RNN", STRUCTURE, REGULARIZATION, DROPOUT, epochs,
                                           rnn_loss, rnn_corr, "", t_rnn_loss, t_rnn_corr, ""]

                    if NN:
                        folder = os.path.join(tax, "NN")
                        res_map = run_NN(X, y, missing_values, params, name, folder, NUMBER_OF_SAMPLES,
                                         None, NUMBER_OF_BACTERIA, person_indexes=person_indexes,
                                         k_fold=k_fold, task_id="all_bact")

                        nn_loss = res_map["TEST"]["loss"]
                        nn_corr = res_map["TEST"]["corr"]
                        nn_r2 = res_map["TEST"]["r2"]

                        t_nn_loss = res_map["TRAIN"]["loss"]
                        t_nn_corr = res_map["TRAIN"]["corr"]
                        t_nn_r2 = res_map["TRAIN"]["r2"]

                        print("loss=" + str(nn_loss))
                        print("corr=" + str(nn_corr))
                        print("r2=" + str(nn_r2))

                        df.loc[len(df)] = ["Multi Bacteria - NN", STRUCTURE, REGULARIZATION, DROPOUT, epochs,
                                           nn_loss, nn_corr, nn_r2, t_nn_loss, t_nn_corr, t_nn_r2]

                        if nni_:
                            if report_loss:
                                nni.report_final_result(nn_loss)
                            elif report_correlation:
                                nni.report_final_result(nn_corr)

                    df.to_csv(title, index=False)


if __name__ == "__main__":  # create an option for nni - get params from file
    run_simulation_data = False
    # best_bacteria_path = "interaction_network_structure_20_fold_test_size_0.5_bacteria_conclusions.csv"
    best_bacteria_path = "run_all_types_of_regression_10_fold_test_size_0.5_bacteria_conclusions.csv"
    if run_simulation_data:
        tax = "Simulations"
        time_serie_path = 'simulation_time_serie_X_y_for_all_bacteria.csv'
        reg_path = 'X_y_for_all_bacteria.csv'

    else:
        tax = os.path.join('GDM', 'tax=5')
        # tax = os.path.join('VitamineA', 'tax=5')
        time_serie_path = 'time_serie_X_y_for_all_bacteria.csv'
        reg_path = 'X_y_for_all_bacteria.csv'

    run_multi_bacteria(tax, time_serie_path, reg_path, NN_or_RNN="RNN", results_df_title="bacteria_grid_search_df")

