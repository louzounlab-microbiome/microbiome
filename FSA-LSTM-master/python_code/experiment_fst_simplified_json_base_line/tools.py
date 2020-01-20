import pickle
import csv

from fst import FST


def read_pkl():
    return pickle.load(open("all_tran.pkl", "rb"))


def create_csv(all_tran):
    table = [["k", "idx", "cycles", "deg", "modularity"]]
    for edges_to_add, list_fst in all_tran.items():
        for i, (fst, stats) in enumerate(list_fst):
            table.append([edges_to_add, i, stats['cycles'], stats['deg'], stats['modularity']])
    with open("fst_more_edges_stats.csv", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(table)
    e = 0


def load_fst_by_nodes_to_add(k):
    all_trans = read_pkl()

    best = [3, 9, 3, 7, 4, 3, 5, 2, 7, 0, 5, 8, 1, 3, 9, 9, 4, 8, 8, 9]
    k_to_best = {k * 10: idx for k, idx in zip(range(1, 21), best)}
    return all_trans[k][k_to_best[k]][0]


if __name__ == '__main__':
    create_csv(read_pkl())
    load_fst_by_nodes_to_add(200)
