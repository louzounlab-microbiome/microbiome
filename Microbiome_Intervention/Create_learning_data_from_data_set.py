from random import shuffle
import pandas as pd
from sklearn.model_selection import LeaveOneOut
import numpy as np
import copy
import os


def remove_none_in_X_y(X, Y):
    """
    Removing missing values from X, y.
    :param X: (DataFrame) features.
    :param Y: (list) tags.
    :return: subset of X, Y pairs such that no 'na' values are found.
    """
    idx_to_keep = []
    for i, (x, y) in enumerate(zip(X, Y)):
        if not (pd.isna(x[0]) or pd.isna(y[0])):
            idx_to_keep.append(i)

    X = X[idx_to_keep]
    Y = [y for i, y in enumerate(Y) if i in idx_to_keep]
    return X, Y


def create_data_for_single_bacteria_model_learning(mapping_file, bacteria, time_point_column,
                                                   id_to_time_point_map, id_to_participant_map, id_to_features_map,
                                                   participant_to_ids_map, time_point_to_index,
                                                   ids_list, features_list, load_and_save_path):
    """
    Create a data frame for each bacterium that has the columns: ['Time Point', 'ID', 'X', 'y']
    X values are all bacteria values in a certain time point, Y values are the change
    in time for a *single* bacteria (relative to the next time point) of the X values.
    Time Point and ID Needed for later.
    :param all params: 'TimeSerieDataSet' class variables.
    :return: (string) name of txt file name holding all the name of the files created for single bacteria learning
    """
    num_of_time_points = len(set(id_to_time_point_map.values()))
    child_to_time_series = {child: [[None]]*num_of_time_points for child in set(id_to_participant_map.values())}
    for child, ids in participant_to_ids_map.items():
        for id in ids:
            p = mapping_file.loc[id, time_point_column]
            if id in id_to_features_map.keys():
                b = id_to_features_map[id]
                child_to_time_series[child][time_point_to_index[p]] = b
            elif str(id) in id_to_features_map.keys():
                b = id_to_features_map[str(id)]
                child_to_time_series[child][time_point_to_index[p]] = b
            else:
                child_to_time_series[child][time_point_to_index[p]] = [None]

    child_to_time_series_df = pd.DataFrame(data=child_to_time_series)

    # create a delta_features for markov model- this is the tags!
    # we would create a model that get as X: the values in time t
    # and as y: the change in a single bacterium in time t+1
    child_to_delta_series = {child: [[None]]*(num_of_time_points-1) for child in set(id_to_participant_map.values())}
    for child, time_serie in child_to_time_series.items():
        for i in range(num_of_time_points-1):
            if type(time_serie[i+1]) == np.ndarray and type(time_serie[i]) == np.ndarray:
                child_to_delta_series[child][i] = time_serie[i+1] - time_serie[i]
    child_to_delta_series_df = pd.DataFrame(data=child_to_delta_series)

    # create a folder for each bacteria, with all time data points data
    bact_list = []
    df_paths_list = []
    for bact_num, bact in enumerate(bacteria):
        df = pd.DataFrame(columns=['Time Point', 'ID', 'X', 'y'])
        for time_point in range(num_of_time_points - 1):
            time_point_X = child_to_time_series_df.loc[time_point].values
            X = copy.deepcopy(time_point_X)
            # get delta change form bacteria-bact num in all samples
            pre_y = child_to_delta_series_df.loc[time_point].values
            Y = []
            for row in pre_y:
                if pd.isna(row[0]):
                    Y.append([None])
                else:
                    Y.append([row[bact_num]])

            # remove entries with None information:
            X, Y = remove_none_in_X_y(X, Y)
            # save to csv file for later
            for x, y in zip(X, Y):
                df.loc[(len(df))] = [time_point, ids_list[features_list.index(list(x))], x.__str__()[2:-1].replace("\n", "").replace("  ", " "), y.__str__()[1:-1]]

        file_name = "X_y_for_bacteria_number_" + str(bact_num) + ".csv"
        df.to_csv(os.path.join(load_and_save_path, file_name), index=False)
        df_paths_list.append(file_name)

    with open(os.path.join(load_and_save_path, "files_names.txt"), "w") as paths_file:
        for path in df_paths_list:
            paths_file.write(path + '\n')
    return "files_names.txt"


def create_data_for_multi_bacteria_model_learning(mapping_file, tax, bacteria, time_point_column,
                                                   id_to_time_point_map, id_to_participant_map, id_to_features_map,
                                                   participant_to_ids_map, time_point_to_index,
                                                   ids_list, features_list, load_and_save_path):
    """
    Create a data frame that has the columns: ['Time Point', 'ID', 'X', 'y']
    X values are all bacteria values in a certain time point, Y values are the change
    in time for *all* bacteria (relative to the next time point) of the X values.
    Time Point and ID Needed for later.
    :param all params: 'TimeSerieDataSet' class variables.
    :return: (string) name of txt file name holding all the name of the files created for single bacteria learning
    """
    num_of_time_points = len(set(id_to_time_point_map.values()))
    child_to_time_series = {child: [[None]]*num_of_time_points for child in set(id_to_participant_map.values())}
    for child, ids in participant_to_ids_map.items():
        for id in ids:
            p = mapping_file.loc[id, time_point_column]
            if id in id_to_features_map.keys():
                b = id_to_features_map[id]
                child_to_time_series[child][time_point_to_index[p]] = b
            elif str(id) in id_to_features_map.keys():
                b = id_to_features_map[str(id)]
                child_to_time_series[child][time_point_to_index[p]] = b
            else:
                child_to_time_series[child][time_point_to_index[p]] = [None]

    # index = time times as range(number of time points)
    # columns = child_number
    child_to_time_series_df = pd.DataFrame(data=child_to_time_series)

    # create a delta_features for markov model- this is the tags!
    # we would create a model that get as X: the values in time t
    # and as y: the change in a single bacterium in time t+1
    child_to_delta_series = {child: [[None]]*(num_of_time_points-1) for child in set(id_to_participant_map.values())}
    for child, time_serie in child_to_time_series.items():
        for i in range(num_of_time_points-1):
            if type(time_serie[i+1]) == np.ndarray and type(time_serie[i]) == np.ndarray:
                child_to_delta_series[child][i] = time_serie[i+1] - time_serie[i]
    # index = time times as range(number of time points)
    # columns = child_number
    child_to_delta_series_df = pd.DataFrame(data=child_to_delta_series)

    # create a folder for each bacteria, with all time data points data
    df = pd.DataFrame(columns=['Time Point', 'ID', 'X', 'y'])
    for time_point in range(num_of_time_points - 1):
        time_point_X = child_to_time_series_df.loc[time_point].values
        X = copy.deepcopy(time_point_X)
        # get delta change form bacteria-bact num in all samples
        pre_y = child_to_delta_series_df.loc[time_point].values
        Y = []
        for row in pre_y:
            if pd.isna(row[0]):
                Y.append([None])
            else:
                Y.append(row)

        # remove entries with None information:
        X, Y = remove_none_in_X_y(X, Y)
        # save to csv file for later
        for x, y in zip(X, Y):
            df.loc[(len(df))] = [time_point, ids_list[features_list.index(list(x))],
                                 x.__str__()[2:-1].replace("\n", "").replace("  ", " "),
                                 y.__str__()[2:-1].replace("\n", "").replace("  ", " ")]

    file_name = "X_y_for_all_bacteria.csv"
    df.to_csv(os.path.join(load_and_save_path, file_name), index=False)

    return "X_y_for_all_bacteria.csv"


def create_data_as_time_serie_for_signal_bacteria_model_learning(mapping_file, tax, bacteria, time_point_column,
                                          id_to_time_point_map, id_to_participant_map, id_to_features_map,
                                          participant_to_ids_map, time_point_to_index, load_and_save_path):
    """
    Create a data frame for each bacterium that has the columns: ['X', 'y']
    X values are a serie all bacteria values in all the time point, Y values are a serie of the change
    in time for a *single* bacteria (relative to the next time point) of the X values.
    :param all params: 'TimeSerieDataSet' class variables.
    :return: (string) name of txt file name holding all the name of the files created for single bacteria learning
    """
    ids_list = []
    features_list = []
    for key, val in id_to_features_map.items():
        ids_list.append(key.split("-")[0])
        features_list.append(list(val))

    num_of_time_points = len(set(id_to_time_point_map.values()))
    num_of_bact_values = max([len(val) for val in id_to_features_map.values()])
    child_to_time_series = {child: [[None]]*num_of_time_points for child in set(id_to_participant_map.values())}
    for child, ids in participant_to_ids_map.items():
        for id in ids:
            p = mapping_file.loc[id, time_point_column]
            if id in id_to_features_map.keys():
                b = id_to_features_map[id]
                child_to_time_series[child][time_point_to_index[p]] = b
            elif str(id) in id_to_features_map.keys():
                b = id_to_features_map[str(id)]
                child_to_time_series[child][time_point_to_index[p]] = b
            else:
                child_to_time_series[child][time_point_to_index[p]] = [None]

    # index = time times as range(number of time points)
    # columns = child_number
    child_to_time_series_df = pd.DataFrame(data=child_to_time_series)

    # create a delta_features for markov model- this is the tags!
    # we would create a model that get as X: the values in all time points shape=(time points-1, bacteria)
    # and as y: the change in a single bacterium in time t+1 for each time point t shape=(time points-1)
    child_to_delta_series = {child: [[None]]*(num_of_time_points-1) for child in set(id_to_participant_map.values())}
    for child, time_serie in child_to_time_series.items():
        for i in range(num_of_time_points-1):
            if type(time_serie[i+1]) == np.ndarray and type(time_serie[i]) == np.ndarray:
                child_to_delta_series[child][i] = time_serie[i+1] - time_serie[i]

    # index = time times as range(number of time points)
    # columns = child_number
    child_to_delta_series_df = pd.DataFrame(data=child_to_delta_series)

    df_paths_list = []
    for bact_num, bact in enumerate(bacteria):
        df = pd.DataFrame(columns=['X', 'y'])
        for col in child_to_time_series_df.columns:
            X = child_to_time_series_df[col][:-1]
            for i, x in enumerate(X):
                if pd.isna(x[0]):
                    X[i] = np.array([-1.0 for n in range(num_of_bact_values)])
            # remove last time point values
            pre_y = child_to_delta_series_df[col]
            Y = []
            for row in pre_y:
                if pd.isna(row[0]):
                    Y.append([-1.0])
                else:
                    Y.append([row[bact_num]])
            # X, Y = remove_none_in_X_y(X, Y)  # need to remove Nones??  ', '.join(mylist)
            # Y = [y[0] for y in Y]
            df.loc[(len(df))] = [';'.join(map(str, X)).replace("\n", "").replace("   ", " ").replace("  ", " "), ';'.join(map(str, Y))]
        file_name = "time_serie_X_y_for_bacteria_number_" + str(bact_num) + ".csv"
        df.to_csv(os.path.join(load_and_save_path, file_name), index=False)
        df_paths_list.append(file_name)

    with open(os.path.join(load_and_save_path, "time_serie_files_names.txt"), "w") as paths_file:
        for path in df_paths_list:
            paths_file.write(path + '\n')
    return "time_serie_files_names.txt"


def create_data_as_time_serie_for_multi_bacteria_model_learning(mapping_file, tax, bacteria, time_point_column,
                                          id_to_time_point_map, id_to_participant_map, id_to_features_map,
                                          participant_to_ids_map, time_point_to_index, load_and_save_path):
    """
    Create a data frame that has the columns: ['X', 'y']
    X values are a serie all bacteria values in all the time point, Y values are a serie of the change
    in time for *all* bacteria (relative to the next time point) of the X values.
    :param all params: 'TimeSerieDataSet' class variables.
    :return: (string) name of txt file name holding all the name of the files created for single bacteria learning
    """
    num_of_time_points = len(set(id_to_time_point_map.values()))
    num_of_bact_values = max([len(val) for val in id_to_features_map.values()])
    child_to_time_series = {child: [[None]]*num_of_time_points for child in set(id_to_participant_map.values())}
    for child, ids in participant_to_ids_map.items():
        for id in ids:
            p = mapping_file.loc[id, time_point_column]
            if id in id_to_features_map.keys():
                b = id_to_features_map[id]
                child_to_time_series[child][time_point_to_index[p]] = b
            elif str(id) in id_to_features_map.keys():
                b = id_to_features_map[str(id)]
                child_to_time_series[child][time_point_to_index[p]] = b
            else:
                child_to_time_series[child][time_point_to_index[p]] = [None]

    # index = time times as range(number of time points)
    # columns = child_number
    child_to_time_series_df = pd.DataFrame(data=child_to_time_series)

    # create a delta_features for markov model- this is the tags!
    # we would create a model that get as X: the values in all time points shape=(time points-1, bacteria)
    # and as y: the change in a single bacterium in time t+1 for each time point t shape=(time points-1)
    child_to_delta_series = {child: [[None]]*(num_of_time_points-1) for child in set(id_to_participant_map.values())}
    for child, time_serie in child_to_time_series.items():
        for i in range(num_of_time_points-1):
            if type(time_serie[i+1]) == np.ndarray and type(time_serie[i]) == np.ndarray:
                child_to_delta_series[child][i] = time_serie[i+1] - time_serie[i]

    # index = time times as range(number of time points)
    # columns = child_number
    child_to_delta_series_df = pd.DataFrame(data=child_to_delta_series)

    df_paths_list = []
    df = pd.DataFrame(columns=['X', 'y'])
    for col in child_to_time_series_df.columns:
        X = child_to_time_series_df[col][:-1]
        for i, x in enumerate(X):
            if pd.isna(x[0]):
                X[i] = np.array([-1.0 for n in range(num_of_bact_values)])
        # remove last time point values
        pre_y = child_to_delta_series_df[col]
        Y = []
        for row in pre_y:
            if pd.isna(row[0]):
                Y.append([-1.0 for n in range(num_of_bact_values)])
            else:
                Y.append(list(row))
        df.loc[(len(df))] = [';'.join(map(str, X)).replace("\n", "").replace("   ", " ").replace("  ", " "), ';'.join(map(str, Y)).replace("\n", "").replace(",", "")]
    file_name = "time_serie_X_y_for_all_bacteria.csv"
    df.to_csv(os.path.join(load_and_save_path, file_name), index=False)
    df_paths_list.append(file_name)

    with open(os.path.join(load_and_save_path, "multi_bacteria_time_serie_files_names.txt"), "w") as paths_file:
        for path in df_paths_list:
            paths_file.write(path + '\n')
    return "multi_bacteria_time_serie_files_names.txt"


def get_adapted_X_y_for_wanted_learning_task(folder, path, task, k_fold=5, test_size=0.3):
    """
    Extract X, y from the different file types from the data generated for the different learning types -
    Regular/TimeSerie, Single Bacteria Prediction/Multi Bacteria Prediction
    :param folder: (string) files folder
    :param path: string) path of file holding the X, y data.
    :param task: string) type of task : "regular" / "multi_bact_regular" / "time_serie" / "multi_bact_time_serie"
    :param k_fold: (int) returns K fold of the split to train and test
    :param test_size: ((float) the size of the test for the split to train and test
    :return: different for each task:
    "regular" = X_trains, X_tests, y_trains, y_tests, name
    "multi_bact_regular" = X, y, person_indexes, name
    "time_serie"/"multi_bact_time_serie" = repaired_X, repaired_Y, missing_values, name
    repaired means that missing values were replaced by the mean, missing_values is an indicator for replaced
    missing values.
    """
    if task == "regular":
        X, y, person_indexes, name = get_X_y_from_file_path(folder, path)
        # train test.................
        # devide to train and test-
        X = np.array(X)
        y = np.array(y)
        X_trains, X_tests, y_trains, y_tests = [], [], [], []
        split_list = [1 - test_size, test_size]
        split_list = np.multiply(np.cumsum(split_list), len(X)).astype("int").tolist()
        # list of shuffled indices to sample randomly

        if type(k_fold) == int:
            for n in range(k_fold):
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

                # list of shuffled indices to sample randomly
                shuffled_idx = []
                shuffle(person_indexes)
                for arr in person_indexes:
                    for val in arr:
                        shuffled_idx.append(val)

                # split the data itself
                X_train = X[shuffled_idx[:split_list[0]]]
                y_train = y[shuffled_idx[:split_list[0]]]

                X_test = X[shuffled_idx[split_list[0]:split_list[1]]]
                y_test = y[shuffled_idx[split_list[0]:split_list[1]]]

                X_trains.append(X_train)
                X_tests.append(X_test)
                y_trains.append(y_train)
                y_tests.append(y_test)

        elif k_fold == "loo":  # hold-one-subject-out cross-validation
            loo = LeaveOneOut()
            for train_index, test_index in loo.split(X):
                X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
                X_trains.append(X_train)
                X_tests.append(X_test)
                y_trains.append(y_train)
                y_tests.append(y_test)
        return X_trains, X_tests, y_trains, y_tests, name

    elif task == "multi_bact_regular":
        X, y, person_indexes, name = get_multi_bact_X_y_from_file_path(folder, path)
        return X, y, person_indexes, name

    elif task == "time_serie":
        return get_time_serie_X_y_from_file_path(folder, path)

    elif task == "multi_bact_time_serie":
        return get_multi_bact_time_serie_X_y_from_file_path(folder, path)


def get_X_y_from_file_path(tax, path):
    df = pd.read_csv(os.path.join(tax, path))
    X = []
    for i, s_x in enumerate(df['X']):
        values = []
        s = s_x.split(" ")
        for val in s:
            if len(val) > 0:
                val = float(val)
                values.append(val)
        X.append(values)

    X = np.array(X)

    y_list = [float(y) for y in df['y'].to_numpy()]


    id_col = df["ID"]
    person_to_indexes = {key:[] for key in list(set(id_col))}
    for idx, id in enumerate(id_col):
        person_to_indexes[id].append(idx)
    person_indexes = list(person_to_indexes.values())

    name = path.split(".")[0]

    return X, y_list, person_indexes, name


def get_multi_bact_X_y_from_file_path(tax, path):
    df = pd.read_csv(os.path.join(tax, path))
    X = []
    for i, s_x in enumerate(df['X']):
        values = []
        s = s_x.split(" ")
        for val in s:
            if len(val) > 0:
                val = float(val)
                values.append(val)
        X.append(values)

    X = np.array(X)

    Y = []
    for i, s_y in enumerate(df['y']):
        values = []
        s = s_y.split(" ")
        for val in s:
            if len(val) > 0:
                val = float(val)
                values.append(val)
        Y.append(values)

    Y = np.array(Y)


    id_col = df["ID"]
    person_to_indexes = {key:[] for key in list(set(id_col))}
    for idx, id in enumerate(id_col):
        person_to_indexes[id].append(idx)
    person_indexes = list(person_to_indexes.values())

    name = path.split(".")[0]

    return X, Y, person_indexes, name


def get_time_serie_X_y_from_file_path(tax, path):
    df = pd.read_csv(os.path.join(tax, path))

    # We will monitor the indexes of the missing values ​​which we will ignore when calculating the loss
    # using the Y dimensions to built matrix - shape = [number of samples, number of time points]
    num_samples = len(df['y'])
    num_time_points = len(df['y'][0].split(";"))
    missing_values = np.full((num_samples, num_time_points), 1)

    X = []
    for sample_i, sample in enumerate(df['X']):
        sample_X = []
        try:
            X_t_0_n = sample.split(";")  # split to X0, X1... Xn
            # for X_i in X_t_0_n, keep time data separate from other time points
            for time_i, x in enumerate(X_t_0_n):
                values = []
                Xt_0_to_m = x.split(" ")  # split to values in Xi
                for val in Xt_0_to_m:
                    val = val.strip("\n").strip("[]")
                    try:
                        if val != "":
                            val = float(val)
                            values.append(val)
                    except ValueError:
                        pass
                sample_X.append(np.array(values))
                if values == [-1.0]*len(values):
                    missing_values[sample_i, time_i] = 0
            X.append(np.array(sample_X))
        except AttributeError:
            print(str(sample_i) + " x " + str(sample))


    Y = []
    for sample_i, sample in enumerate(df['y']):
        sample_y = []
        try:
            y_t_0_n = sample.split(";")  # split to y0, y1... yn
            for time_i, y in enumerate(y_t_0_n):
                val = y.strip("[]")
                try:
                    val = float(val)
                    sample_y.append(val)
                    if val == -1.0:
                        missing_values[sample_i, time_i] = 0
                except ValueError:
                    # We will monitor the indexes of the missing values ​​which we will ignore when calculating the loss
                    sample_y.append(0.0)
                    missing_values[sample_i, time_i] = 0

            Y.append(sample_y)
        except AttributeError:
            print(str(sample_i) + " y " + str(sample))

    X = np.array(X, ndmin=3)
    Y = np.array(Y)
    name = path.split(".")[0]

    samples_list = []
    for row in X:
        for sample in row:
            if list(sample) != [-1.0] * X.shape[2]:
                  samples_list.append(sample)

    bacteria_mean_list = np.mean(np.array(samples_list), axis=0)
    repaired_X = np.where(X == [-1.0] * X.shape[2], bacteria_mean_list, X)

    repaired_Y = copy.deepcopy(Y)
    for t_i, times in enumerate(repaired_Y):
        for v_i, val in enumerate(times):
            if val == -1.0:
                repaired_Y[t_i][v_i] = bacteria_mean_list[v_i]

    return repaired_X, repaired_Y, missing_values, name


def get_multi_bact_time_serie_X_y_from_file_path(tax, path):
    df = pd.read_csv(os.path.join(tax, path))

    # We will monitor the indexes of the missing values ​​which we will ignore when calculating the loss
    # using the Y dimensions to built matrix - shape = [number of samples, number of time points]
    num_samples = len(df['y'])
    num_time_points = len(df['y'][0].split(";"))
    missing_values = np.full((num_samples, num_time_points), 1)

    X = []
    for sample_i, sample in enumerate(df['X']):
        sample_X = []
        X_t_0_n = sample.split(";")  # split to X0, X1... Xn
        # for X_i in X_t_0_n, keep time data separate from other time points
        for x_i, x in enumerate(X_t_0_n):
            values = []
            Xt_0_to_m = x.split(" ")  # split to values in Xi
            for val in Xt_0_to_m:
                val = val.strip("\n").strip("[]")
                try:
                    if val != "":
                        val = float(val)
                        values.append(val)
                        if val == -1.0:
                            missing_values[sample_i, x_i] = 0

                except ValueError:
                    pass
            sample_X.append(np.array(values))
        X.append(np.array(sample_X, ndmin=2))


    Y = []
    for sample_i, sample in enumerate(df['y']):
        sample_y = []
        y_t_0_n = sample.split(";")  # split to y0, y1... yn
        for y_i, y in enumerate(y_t_0_n):
            values = []
            yt_0_to_m = y.split(" ")  # split to values in Xi
            for val in yt_0_to_m:
                val = val.strip("\n").strip("[]")
                try:
                    if val != "":
                        val = float(val)
                        values.append(val)
                        if val == -1.0:
                            missing_values[sample_i, y_i] = 0
                except ValueError:
                    pass
            sample_y.append(np.array(values))
        Y.append(np.array(sample_y, ndmin=2))

    X = np.array(X, ndmin=3)
    Y = np.array(Y, ndmin=3)
    name = path.split(".")[0]

    samples_list = []
    for row in X:
        for sample in row:
            if list(sample) != [-1.0] * X.shape[2]:
                  samples_list.append(sample)

    bacteria_mean_list = np.mean(np.array(samples_list), axis=0)
    repaired_X = np.where(X == [-1.0] * X.shape[2], bacteria_mean_list, X)

    repaired_Y = copy.deepcopy(Y)
    for t_i, times in enumerate(repaired_Y):
        for vals in times:
            for v_i, val in enumerate(vals):
                if val == -1.0:
                    repaired_Y[t_i][v_i] = bacteria_mean_list[v_i]



    return repaired_X, repaired_Y, missing_values, name


