from LearningMethods.Regression.regressors import *
import numpy as np
import copy


def remove_none_in_X_y(X, Y):
    idx_to_remove = []
    idx_to_keep = []
    for i, (x, y) in enumerate(zip(X, Y)):
        if pd.isna(x[0]) or pd.isna(y[0]):
            idx_to_remove.append(i)
        else:
            idx_to_keep.append(i)
    X = X[idx_to_keep]
    Y = [y for i, y in enumerate(Y) if i in idx_to_keep]
    # X = np.delete(X, idx_to_remove, axis=0)
    # Y = np.delete(Y, idx_to_remove, axis=0)
    return X, Y


def create_data_for_signal_bacteria_model_learning(OtuMf, tax, bacteria, time_point_column,
                                          id_to_time_point_map, id_to_participant_map, id_to_features_map,
                                          participant_to_ids_map, time_point_to_index):
    num_of_time_points = len(set(id_to_time_point_map.values()))
    child_to_time_series = {child: [[None]]*num_of_time_points for child in set(id_to_participant_map.values())}
    for child, ids in participant_to_ids_map.items():
        for id in ids:
            p = OtuMf.mapping_file.loc[id, time_point_column]
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
    bact_list = []
    df_paths_list = []
    for bact_num, bact in enumerate(bacteria):
        df = pd.DataFrame(columns=['Time Point', 'X', 'y'])
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
                df.loc[(len(df))] = [time_point, x, y]

        file_name = "X_y_for_bacteria_number_" + str(bact_num) + ".csv"
        df.to_csv(os.path.join(tax, file_name), index=False)
        df_paths_list.append(file_name)

    with open(os.path.join(tax, "files_names.txt"), "w") as paths_file:
        for path in df_paths_list:
            paths_file.write(path + '\n')
    return bact_list, "files_names.txt"


def create_data_as_time_serie_for_signal_bacteria_model_learning(OtuMf, tax, bacteria, time_point_column,
                                          id_to_time_point_map, id_to_participant_map, id_to_features_map,
                                          participant_to_ids_map, time_point_to_index):
    num_of_time_points = len(set(id_to_time_point_map.values()))
    child_to_time_series = {child: [[None]]*num_of_time_points for child in set(id_to_participant_map.values())}
    for child, ids in participant_to_ids_map.items():
        for id in ids:
            p = OtuMf.mapping_file.loc[id, time_point_column]
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
            X = child_to_time_series_df[col][:-1]  # remove last time point values
            pre_y = child_to_delta_series_df[col]
            Y = []
            for row in pre_y:
                if pd.isna(row[0]):
                    Y.append([None])
                else:
                    Y.append([row[bact_num]])
            X, Y = remove_none_in_X_y(X, Y)  # need to remove Nones??  ', '.join(mylist)
            Y = [y[0] for y in Y]
            df.loc[(len(df))] = [';'.join(map(str, X)).replace("\n", "").replace("   ", " ").replace("  ", " "), ';'.join(map(str, Y))]
        file_name = "time_serie_X_y_for_bacteria_number_" + str(bact_num) + ".csv"
        df.to_csv(os.path.join(tax, file_name), index=False)
        df_paths_list.append(file_name)

    with open(os.path.join(tax, "time_serie_files_names.txt"), "w") as paths_file:
        for path in df_paths_list:
            paths_file.write(path + '\n')
    return "time_serie_files_names.txt"


def create_data_for_markob_model_learning(OtuMf, tax, bacteria, time_point_column,
                                          id_to_time_point_map, id_to_participant_map, id_to_features_map,
                                          participant_to_ids_map, time_point_to_index):
    num_of_time_points = len(set(id_to_time_point_map.values()))
    child_to_time_series = {child: [[None]]*num_of_time_points for child in set(id_to_participant_map.values())}
    for child, ids in participant_to_ids_map.items():
        for id in ids:
            p = OtuMf.mapping_file.loc[id, time_point_column]
            bacteria = id_to_features_map[id]
            child_to_time_series[child][time_point_to_index[p]] = bacteria
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

    time_points_list = []
    for time_point in range(num_of_time_points-1):
        time_point_folder = 'time=' + str(time_point)
        time_points_list.append(time_point_folder)
        if not os.path.exists(os.path.join(tax, time_point_folder)):
            os.mkdir(os.path.join(tax, time_point_folder))
        ids = list(child_to_time_series_df.columns)
        time_point_X = child_to_time_series_df.loc[time_point].values
        df_paths_list = []
        for bact_num, bact in enumerate(bacteria):
            X = copy.deepcopy(time_point_X)
            # get delta change form bacteria-bact num in all samples
            pre_y = child_to_delta_series_df.loc[time_point]
            y = []
            for row in pre_y:
                if pd.isna(row[0]):
                    y.append([None])
                else:
                    y.append([row[bact_num]])

            # remove entries with None information:
            X, y = remove_none_in_X_y(X, y)
            # save to csv file for later
            df = pd.DataFrame(columns=['X', 'y'], index=range(len(X)))
            df['X'] = X
            df['y'] = y
            file_name = "X_y_" + str(bact_num) + ".csv"
            df.to_csv(os.path.join(tax, time_point_folder, file_name), index=False)
            df_paths_list.append(file_name)

        with open(os.path.join(tax, time_point_folder, "files_names.txt"), "w") as paths_file:
            for path in df_paths_list:
                paths_file.write(path + '\n')
    return time_points_list, "files_names.txt"


def get_adapted_X_y_for_wanted_learning_task(tax, path, task, k_fold=5, test_size=0.2):
    if task == "regular":
        X, y, name = get_X_y_from_file_path(tax, path)
        # train test.................
        # devide to train and test-
        X = np.array(X)
        y = np.array(y)
        X_trains, X_tests, y_trains, y_tests = [], [], [], []
        if type(k_fold) == int:
            for n in range(k_fold):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
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

    elif task == "hidden_measurements":
        time_point_to_hide = [4, 8, 12, 16, 20, 24]
        return get_hidden_measurements_train_and_test_X_y_from_file_path(tax, path, time_point_to_hide)
    elif task == "time_serie":
        return get_time_serie_X_y_from_file_path(tax, path)


def get_X_y_from_file_path(tax, path):
    df = pd.read_csv(os.path.join(tax, path))
    X = []
    for s_x in df['X']:
        values = []
        s = s_x.split(" ")
        for val in s:
            val = val.strip("\n").strip("]").strip("[")
            if len(val) > 0:
                val = float(val)
                values.append(val)
        X.append(values)

    X = np.array(X)
    Y = df['y'].to_numpy()
    y_list = []
    for y in Y:
        y_list.append(float(y[1:-1]))
    name = path.split(".")[0]
    return X, y_list, name


def get_time_serie_X_y_from_file_path(tax, path):
    df = pd.read_csv(os.path.join(tax, path))
    X = []
    for sample in df['X']:
        sample_X = []
        X_t_0_n = sample.split(";")  # split to X0, X1... Xn
        # for X_i in X_t_0_n, keep time data separate from other time points
        for smaple_i, x in enumerate(X_t_0_n):
            # print(str(smaple_i) + "!")
            t = 1
            values = []
            Xt_0_to_m = x.split(" ")  # split to values in Xi
            for val in Xt_0_to_m:
                val = val.strip("\n").strip("[]")
                try:
                    val = float(val)
                    values.append(val)
                    # print(str(t) + ") " + str(val))
                    t += 1
                except ValueError:
                    pass
            sample_X.append(values)
        X.append(sample_X)

    Y = []
    for sample in df['y']:
        sample_y = []
        y_t_0_n = sample.split(";")  # split to y0, y1... yn
        for smaple_i, y in enumerate(y_t_0_n):
            val = y.strip("[]")
            try:
                val = float(val)
                sample_y.append(val)
            except ValueError:
                pass
        Y.append(sample_y)

    X = np.array(X)
    Y = np.array(Y)
    name = path.split(".")[0]
    return X, Y, name


def get_hidden_measurements_train_and_test_X_y_from_file_path(tax, path, time_point_to_hide):
    # We want to remove 6 time points from the learning data of the model in order to forecast them
    # This means that 6 time points will not enter the training set
    # not as a predicted value in time t + 1 (y), nor as a value predicted by in time t (X).
    # Those samples containing a removed time point as X/y will be the test set
    # when separating samples, remove from df, time c, c+1 for all c time point in chosen to remove time points
    df = pd.read_csv(os.path.join(tax, path))
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for t, s_x, s_y in zip(df['Time Point'], df['X'], df['y'].to_numpy()):
        values = []
        s = s_x.split(" ")
        for val in s:
            val = val.strip("\n").strip("]").strip("[")
            if len(val) > 0:
                val = float(val)
                values.append(val)
        if t not in time_point_to_hide and t + 1 not in time_point_to_hide:
            X_train.append(values)
            y_train.append(float(s_y[1:-1]))
        else:
            X_test.append(values)
            y_test.append(float(s_y[1:-1]))
    time_points_str = ""
    for t in time_point_to_hide:
        time_points_str += str(t) + "_"
    name = time_points_str[:-1] + "_" + path.split(".")[0]
    return X_train, X_test, y_train, y_test, name

