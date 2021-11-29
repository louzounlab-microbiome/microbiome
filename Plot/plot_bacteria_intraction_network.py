import pickle
import pandas as pd
from pyvis.network import Network
import matplotlib.colors as plt_clr
import matplotlib.lines as mlines
from pylab import *
import os
from LearningMethods.general_functions import shorten_single_bact_name


def rgbA_colors_generator():
    r = sample(1)
    g = sample(1)
    b = sample(1)
    return (r[0], g[0], b[0])


# color must be in HEX!!!
def get_nodes_colors_by_bacteria_tax_level(bacteria, G_name, taxnomy_level, folder):
    bacteria = [b.split(";") for b in bacteria]
    s = pd.Series(bacteria)
    # get the taxonomy type for the wanted level
    taxonomy_reduced = s.map(lambda x: ';'.join(x[:taxnomy_level]))
    tax_groups = list(set(taxonomy_reduced))  # get the unique taxonomy types
    tax_groups.sort()  # sort for consistency of the color + shape of a group in multiple runs (who has the same groups)
    number_of_tax_groups = len(tax_groups)

    tax_to_color_and_shape_map = {}
    colors = ['#CD1414', '#EE831E', '#F0E31E', '#91F01E', '#1EF08D', '#1EDAF0', '#1E4EF0', '#671EF0', '#F01EE6', '#F01E5A',
              '#773326', '#695B18', '#787A6E', '#040404', '#1C440E', '#ABE0D6', '#ABBDE0', '#DCBFEC', '#ECBFD7', '#0400FF']
    markers = ['D', 'o', '*', '^', 'v']
    shapes = ["diamond", "dot", "star", "triangle", "triangleDown"]
    shape_to_marker_map = {shapes[i]: markers[i] for i in range(len(markers))}

    for i in range(number_of_tax_groups):
        # c = plt_clr.rgb2hex(rgbA_colors_generator())  # random color
        tax_to_color_and_shape_map[tax_groups[i]] = (colors[i%len(colors)], shapes[i%len(shapes)])  # color + shape from list

    color_list = [tax_to_color_and_shape_map[t][0] for t in taxonomy_reduced]
    shape_list = [tax_to_color_and_shape_map[t][1] for t in taxonomy_reduced]
    group_list = [shorten_single_bact_name(t) for t in taxonomy_reduced]

    # create the legend useing matplotlib because pyvis doesn't have legends
    lines = []
    for (key, val) in tax_to_color_and_shape_map.items():
        color = val[0]
        marker = shape_to_marker_map[val[1]]
        line = mlines.Line2D([], [], color=color, marker=marker,
                                  markersize=15, label=key)
        lines.append(line)
    plt.legend(handles=lines, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad = 0.)
    plt.savefig(os.path.join(folder, G_name + "_legend.svg"), bbox_inches='tight', format='svg')
    # plt.show()
    return color_list, shape_list, group_list, tax_to_color_and_shape_map



def plot_bacteria_intraction_network(bacteria, node_list, node_size_list, edge_list, color_list, G_name, folder,
                                     color_by_tax_level=2, directed_G=True, control_color_and_shape=True):
    # create the results folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    nodes_colors, nodes_shapes, group_list, tax_to_color_and_shape_map = \
        get_nodes_colors_by_bacteria_tax_level(bacteria, G_name, taxnomy_level=color_by_tax_level, folder=folder)

    bact_short_names = [shorten_single_bact_name(b) for b in bacteria]
    nodes_with_edges = np.unique(np.array(edge_list).flatten()).tolist()

    net = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=directed_G)
    #net.barnes_hut(gravity=-120000)
    net.force_atlas_2based()

    # for the nodes: you can use either only the group option the automatically colors the group in different colors
    # shaped like dots - no control, or use color and shape to make it you own, in this case group is irrelevant
    for i, node in enumerate(node_list):
        if node in nodes_with_edges:
            if control_color_and_shape:
                net.add_node(node, label=bact_short_names[i], color=nodes_colors[i], value=node_size_list[i],
                            shape=nodes_shapes[i], group=group_list[i])
            else:
                net.add_node(node, label=bact_short_names[i],  value=node_size_list[i],
                            group=group_list[i])

    # for the edges, the colors are what you decide and send
    for i, (u, v) in enumerate(edge_list):
        net.add_edge(u, v, color=color_list[i])


    net.save_graph(os.path.join(folder, G_name + ".html"))
    # net.show(G_name + ".html")


if __name__ == "__main__":
    node_list = pickle.load(open("node_list_0.pkl", "rb"))
    edge_list = pickle.load(open("edge_list_0.pkl", "rb"))
    color_list = pickle.load(open("color_list_0.pkl", "rb"))

    with open("bacteria.txt", "r") as b_file:
        bacteria = b_file.readlines()
        bacteria = [b.rstrip() for b in bacteria]
    # set the size of the nodes, can control it  if wanted
    v = [100] * len(bacteria)

    plot_bacteria_intraction_network(bacteria, node_list, v, edge_list, color_list,
                                     "example_graph", "bacteria_interaction_network")

