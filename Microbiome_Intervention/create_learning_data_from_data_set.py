from LearningMethods.Regression.regressors import *
import numpy as np
import copy

def remove_none_in_X_y(X, Y):
    idx_to_remove = []
    for i, (x, y) in enumerate(zip(X, Y)):
        if pd.isna(x[0]) or pd.isna(y[0]):
            idx_to_remove.append(i)
    X = np.delete(X, idx_to_remove, axis=0)
    Y = np.delete(Y, idx_to_remove, axis=0)
    return X, Y


def create_data_for_signal_bacteria_model_learning(OtuMf, tax, bacteria, time_point_column,
                                          id_to_time_point_map, id_to_participant_map, id_to_features_map,
                                          participant_to_ids_map, time_point_to_index):
    num_of_time_points = len(set(id_to_time_point_map.values()))
    child_to_time_series = {child: [[None]]*num_of_time_points for child in set(id_to_participant_map.values())}
    for child, ids in participant_to_ids_map.items():
        for id in ids:
            p = OtuMf.mapping_file.loc[id, time_point_column]
            if str(id) in id_to_features_map.keys():
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

