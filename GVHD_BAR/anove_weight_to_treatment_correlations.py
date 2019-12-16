import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os


def get_organize_data_for_anova(file):
    # get data
    df = pd.read_csv(file)
    ids = df['id']
    groups = df['group']
    chemotherapy = df['chemotherapy']
    weights_df = df[['weight - 24.10.19', 'weight - 27.10.19', 'weight - 29.10.19', 'weight - 3.11.19', 'weight - 4.11.19',
                     'weight - 5.11.19', 'weight - 6.11.19', 'weight - 7.11.19']]
    weights_df.index = ids
    id_to_chemotherapy_map = {}
    id_to_group_map = {}
    id_to_weights_map = {}
    weights_mat = weights_df.to_numpy()
    initial_weights = [row[0] for row in weights_mat]
    weights_mat = [[item -initial_weights[i] for item in row] for i, row in enumerate(weights_mat)]
    weights_mat = [row[1:] for row in weights_mat]
    time_points = ['weight - 27.10.19', 'weight - 29.10.19', 'weight - 3.11.19', 'weight - 4.11.19',
                     'weight - 5.11.19', 'weight - 6.11.19', 'weight - 7.11.19']

    df = pd.DataFrame(columns=["id", "value", "chemotherapy", "group", "time"])

    for id, chemo, group, weights in zip(ids, chemotherapy, groups, weights_mat):
        id_to_chemotherapy_map[id] = chemo
        id_to_group_map[id] = group
        id_to_weights_map[id] = weights
        for time, w in enumerate(weights):
            df.loc[len(df)] = [id, w, chemo, group, time_points[time]]
    df = df.set_index(df["id"])
    df = df.drop(["id"], axis=1)

    return id_to_chemotherapy_map, id_to_group_map, id_to_weights_map, df


if __name__ == "__main__":
    folder = "WEIGHTS"
    os.chdir(folder)

    file = "weight.csv"

    id_to_chemotherapy_map, id_to_group_map, id_to_weights_map, value_chemo_group_time_df = get_organize_data_for_anova(file)
    value_chemo_group_time_df.to_csv("value_chemo_group_time_df.csv")

    # anova test
    model = ols('value ~ C(group) + C(time) + C(chemotherapy) + '
                'C(group):C(time) + C(group):C(chemotherapy) + '
                'C(chemotherapy):C(time) + C(chemotherapy):C(time): C(group)',
                data=value_chemo_group_time_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)

    p_val_map = {}

    p_val_map["group"] = anova_table["PR(>F)"]["C(group)"]
    p_val_map["time"] = anova_table["PR(>F)"]["C(time)"]
    p_val_map["chemotherapy"] = anova_table["PR(>F)"]["C(chemotherapy)"]

    p_val_map["group*time"] = anova_table["PR(>F)"]["C(group):C(time)"]
    p_val_map["group*chemotherapy"] = anova_table["PR(>F)"]["C(group):C(chemotherapy)"]
    p_val_map["chemotherapy*time"] = anova_table["PR(>F)"]["C(chemotherapy):C(time)"]
    p_val_map["chemotherapy*time*group"] = anova_table["PR(>F)"]["C(chemotherapy):C(time):C(group)"]

    with open("anova_p_value_results.txt", "w") as file:
        file.write("ANOVE 3 way test\n")
        for key, val in p_val_map.items():
            file.write("p value of " + key + ": " + str(val) + "\n")
    print("a")
