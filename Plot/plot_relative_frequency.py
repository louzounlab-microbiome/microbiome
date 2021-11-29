import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_rel_freq(df: pd.DataFrame, folder=None, taxonomy_level=3):
    # df = easy_otu_name(df)
    df = df.reindex(df.mean().sort_values().index, axis=1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax2.axis('off')
    df.plot.bar(stacked=True, ax=ax, width=1.0, colormap='Spectral')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="x-small")
    ax.xaxis.set_ticks([])
    ax.set_xlabel("")
    ax.set_title("Relative frequency with taxonomy level " + str(taxonomy_level))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # plt.show()
    plt.savefig(f"{folder}/relative_frequency_stacked.png")


def easy_otu_name(df):
    df.columns = [str([h[4:].capitalize() for h in i.split(";")][-2:])
                      .replace("[", "").replace("]", "").replace("\'", "") for i in df.columns]
    return df
