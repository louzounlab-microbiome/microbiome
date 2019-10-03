import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# plot significant bacteria. color each significant - time / state/ time*state in different color
def plot_anove_significant_bacteria(colors_to_label_dict, bacteria, p_values, p_values_colors, p_0, taxnomy_level, folder=None, round=""):
    # extract the last meaningful name - long multi level names to the lowest level definition
    short_bacteria_names = []
    for f in bacteria:
        i = 1
        while len(f.split(";")[-i]) < 5:  # meaningless name
            i += 1
        short_bacteria_names.append(f.split(";")[-i])

    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(bacteria))
    ax.barh(y_pos, p_values, color=p_values_colors, tick_label=p_values) # palette=color_dict
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_bacteria_names)

    patches = []
    for key in colors_to_label_dict:
        patches.append(mpatches.Patch(color=colors_to_label_dict[key], label=key))
    ax.legend(handles=patches, fontsize=12, loc='lower right')

    plt.yticks(fontsize=10)
    plt.title("Two-way ANOVA - significant bacteria\nTaxnomy level " + str(taxnomy_level) + " (p=" + str(p_0) + ")", fontsize=17)
    ax.set_xlabel("p value", fontsize=12)
    fig.subplots_adjust(left=left_padding)
    if folder:
        plt.savefig(os.path.join(folder, "Two_way_ANOVA_plot_Taxnomy level " + str(taxnomy_level) + "_p=" + str(p_0) + "#" + str(round) + ".svg"), bbox_inches='tight', format='svg')
    else:
        plt.savefig("Two_way_ANOVA_plot_Taxnomy_level_" + str(taxnomy_level) + "_p=" + str(p_0) + "#" + str(round) + ".svg", bbox_inches='tight', format='svg')
    plt.show()


if __name__ == "__main__":
    bacteria = ["aaaaaaaaaaa;", "bbbbbbbbbb;", "ccccccccc;", "ddddddddd;", "eeeeeeee;", "fffffffff;"]
    p_values = [0.003, 0.01, 0.02, 0.05, 0.087, 0.14]
    p_values_colors = ["red", "green", "blue", "green", "blue", "red"]
    anove_labels = ["time", "state", "time*state"]
    anove_labels_colors = ["b", "g", "r"]
    colors_to_label_dict = {key: val for key, val in zip(anove_labels, anove_labels_colors)}
    plot_anove_significant_bacteria(colors_to_label_dict, bacteria, p_values, p_values_colors)