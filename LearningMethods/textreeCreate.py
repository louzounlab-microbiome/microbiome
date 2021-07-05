#from openpyxl import Workbook, load_workbook
import re
import math
import pandas
import networkx as nx
import pickle



"""
every bacteria is an object to easily store it's information
"""
class Bacteria:
    def __init__(self, string, val):
        lst = re.split("; |__| ", string)
        self.val = val
        #removing letters and blank spaces
        for i in range(0, len(lst)):
            if len(lst[i]) < 2:
                lst[i] = 0
        lst = [value for value in lst if value != 0]
        self.lst = lst


"""
manual layout code, not needed
def polarToCar(radius, angle):
    return (radius*math.cos(angle), radius*math.sin(angle))

def sorter(graph, pos):
    for i in range(0,6):
        temp = []
        for node in pos[i]:
            for elem in graph.neighbors(node):
                if elem in pos[i+1] and elem not in temp:
                    temp.append(elem)
        pos[i+1] = temp
    pos[0] = list(pos[0])

def manual_layout(graph, pos):
    sorter(graph, pos)
    anglelayout = {}
    layout = {}
    theta = 2* math.pi/len(pos[6])
    for i in range (0, len(pos[6])):
        layout[pos[6][i]] = polarToCar(6, theta * i)
        anglelayout[pos[6][i]] = theta * i
    for i in range (5, 0, -1):
        for node in pos[i]:
             avg = 0
             count = 0
             for n in graph.neighbors(node):
                 if n in anglelayout:
                    avg += anglelayout[n]
                    count +=1
             if avg != 0 and i > 2:
                avg /= count
                layout[node] = polarToCar(i,avg)
                anglelayout[node] = avg
             
                          else:
                layout[node] = (0,0)
             
    for i in pos[0]:
        layout[i] = (0,0)
    return layout

"""
def create_tax_tree(series, zeroflag=False):
    tempGraph = nx.Graph()
    """workbook = load_workbook(filename="random_Otus.xlsx")
    sheet = workbook.active"""
    valdict = {}
    bac = []
    for i, (tax, val) in enumerate(series.items()):
        # adding the bacteria in every column
        bac.append(Bacteria(tax, val))
        # connecting to the root of the tempGraph
        tempGraph.add_edge("anaerobe", bac[i].lst[0])
        # connecting all levels of the taxonomy
        for j in range(0, len(bac[i].lst) - 1):
            updateval(tempGraph, bac[i], valdict, j, True)
        # adding the value of the last node in the chain
        updateval(tempGraph, bac[i], valdict, len(bac[i].lst) - 1, False)
    valdict["anaerobe"] = valdict["Bacteria"] + valdict["Archaea"]
    return create_final_graph(tempGraph, valdict, zeroflag)


def updateval(graph, bac, vald, num, adde):
    if adde:
        graph.add_edge(bac.lst[num], bac.lst[num + 1])
    # adding the value of the nodes
    if bac.lst[num] in vald:
        vald[bac.lst[num]] += bac.val
    else:
        vald[bac.lst[num]] = bac.val


def create_final_graph(tempGraph, valdict, zeroflag):
    graph = nx.Graph()
    for e in tempGraph.edges():
        if not zeroflag or valdict[e[0]] * valdict[e[1]] != 0:
            graph.add_edge((e[0], valdict[e[0]]),
                           (e[1], valdict[e[1]]))
    return graph


