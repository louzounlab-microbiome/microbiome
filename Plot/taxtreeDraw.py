from LearningMethods.textreeCreate import create_tax_tree
import networkx as nx
import pickle
#import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


def draw_tree(graph):
    labelg = {}
    labelr = {}
    colormap = []
    sizemap = []
    for node in graph:
        if node[0] == "base":
            colormap.append("white")
            sizemap.append(0)
        else:
            if node[1] < 0:
                colormap.append("red")
                labelr[node] = node
            elif node[1] > 0:
                colormap.append("green")
                labelg[node] = node
            else:
                colormap.append("yellow")
            sizemap.append(node[1] / 100 + 5)
    # drawing the graph
    #pos = graphviz_layout(graph, prog="twopi", root="base")
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_size=sizemap, node_color=colormap, width=0.3)
    nx.nx.draw_networkx_labels(graph, pos, labelr, font_size=7, font_color="red")
    nx.nx.draw_networkx_labels(graph, pos, labelg, font_size=7, font_color="green")
    plt.draw()
    plt.savefig("taxtree.png")

