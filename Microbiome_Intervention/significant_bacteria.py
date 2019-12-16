import os
import pandas as pd
import numpy as np

from LearningMethods import shorten_single_bact_name
from Plot import plot_heat_map_from_df, plot_cluster_heat_map_from_df


def check_if_bacteria_correlation_is_significant(df_path, algo):
    # see if the real rho value is 2 std steps from the mixed rho mean, if so -> rho value is significant
    significant_bacteria = []
    df = pd.read_csv(df_path)
    sub_df = df[df["ALGORITHM"] == algo]
    bacteria = sub_df["BACTERIA"]
    rhos = sub_df["RHO"]
    mixed_rho_mean = sub_df["RANDOM_RHO"].mean()
    mixed_rho_std = sub_df["RANDOM_RHO"].std()
    for bact, r in zip(bacteria, rhos):
        if r > mixed_rho_mean + (2*mixed_rho_std) or r < mixed_rho_mean - (2*mixed_rho_std):
            significant_bacteria.append(bact)

    return significant_bacteria


def get_significant_beta_from_file(df_path, algo, significant_bacteria, folder, plot_hist=True):
    important_bacteria_df = pd.DataFrame(columns=["MODEL BACTERIA", "SIGNIFICANT BACTERIA", 'ALGORITHM',
                                                  'BETA MEAN', 'BETA STD', 'BETA'])
    df = pd.read_csv(df_path)

    sub_df = df[df["ALGORITHM"] == algo]
    bacteria = list(sub_df["BACTERIA"])
    beta_list = [beta.split(";") for beta in sub_df["BETA"]]

    all_rhos_df = pd.DataFrame(index=bacteria, columns=bacteria)

    for row_i, beta in enumerate(beta_list):  # bacteria and all her rhos
            if not beta == [' ']:
                numeric_beta = [float(b) for b in beta]
                all_rhos_df.loc[bacteria[row_i]] = numeric_beta

                if bacteria[row_i] in significant_bacteria:
                    mean = np.mean(numeric_beta)
                    std = np.std(numeric_beta)
                    for b_i, b in enumerate(numeric_beta):  # iterate rhos
                        if b > mean + (2 * std) or b < mean - (2 * std):
                            important_bacteria_df.loc[len(important_bacteria_df)] =\
                                [bacteria[row_i], bacteria[b_i], algo, mean, std, b]

    if not os.path.exists(folder):
        os.mkdir(folder)
    all_rhos_df.to_csv(os.path.join(folder, "all_bacteria_rhos_" + algo.replace(" ", "_") + ".csv"))
    important_bacteria_df.to_csv(os.path.join(folder, "important_bacteria_" + algo.replace(" ", "_") + ".csv"))

    # plot heat map of all_rhos_df
    all_rhos_df.index = [shorten_single_bact_name(bact) for bact in all_rhos_df.index]
    all_rhos_df.columns = [shorten_single_bact_name(bact) for bact in all_rhos_df.columns]
    all_rhos_df = all_rhos_df.dropna()
    if plot_hist:
        plot_heat_map_from_df(all_rhos_df, folder.replace("/", " ").replace("_", " ") +
                              "\nBacteria Correlation to The Change in Bacteria Over Time", "Bacteria correlation",
                              "Changing bacteria", folder)

        plot_cluster_heat_map_from_df(all_rhos_df, folder.replace("/", " ").replace("_", " ") +
                              "\nBacteria Clustered Correlation\nChange in Bacteria Over Time", "Bacteria correlation",
                              "Changing bacteria", folder)


if __name__ == "__main__":
    for data_set in ["Diet_study", "Bifidobacterium_bifidum", "VitamineA"]:  # "Diet_study", "Bifidobacterium_bifidum",
        for tax_level in ["tax=4", "tax=5"]:
            for algo in ["ard regression"]:  # "random forest" has no coefficents
                folder = os.path.join(data_set, tax_level, "Significant_bacteria")
                results_df_path = os.path.join(data_set, tax_level, "all_times_all_bacteria_all_models_results_df.csv")
                significant_bacteria = check_if_bacteria_correlation_is_significant(results_df_path, algo)
                get_significant_beta_from_file(results_df_path, algo, significant_bacteria, folder)



