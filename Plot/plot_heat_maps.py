from os.path import join
from matplotlib import pyplot as plt
import seaborn as sns


def plot_heat_map_from_df(data, title, x_label, y_label, folder, pos_neg=True):
  # plot
  fig, ax = plt.subplots(figsize=(7, 6))
  font_size = 8
  plt.yticks(fontsize=font_size)
  plt.xticks(fontsize=font_size)
  if pos_neg:
    # values_range = max(abs(data.astype(float).min().min()), data.astype(float).max().max())
    ax = sns.heatmap(data.astype(float), cmap='RdBu', ax=ax, vmin=-1, vmax=1)
  else:
    ax = sns.heatmap(data.astype(float), cmap='Blues', ax=ax)

  plt.title(title, fontsize=font_size + 5)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.savefig(join(folder, title.replace(" ", "_").replace("\n", "_") + ".svg"), bbox_inches='tight', format='svg')
  plt.show()


def plot_cluster_heat_map_from_df(data, title, x_label, y_label, folder, pos_neg=True, font_size=8):
  # plot
  fig, ax = plt.subplots()  # figsize=(15, 5)
  ax.set_title(title)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  # plt.yticks(fontsize=font_size)
  # plt.xticks(fontsize=font_size)
  if pos_neg:
    # values_range = max(abs(data.astype(float).min().min()), data.astype(float).max().max())
    ax = sns.clustermap(data.astype(float), cmap='RdBu', vmin=-1, vmax=1, metric="correlation")
  else:
    ax = sns.clustermap(data.astype(float), cmap='Blues')

  plt.title(title, fontsize=font_size + 5)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.savefig(join(folder, title.replace(" ", "_").replace("\n", "_") + ".svg"), bbox_inches='tight', format='svg')
  plt.show()