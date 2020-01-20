import pickle
from copy import deepcopy
from random import choice, shuffle

from experiment_fst_simplified_json_base_line.simplified_json_fst_params import SIMPLIFIED_JSON_TRANSITIONS, \
    SIMPLIFIED_JSON_FST, SIMPLIFIED_JSON_INIT_STATE, SIMPLIFIED_JSON_ACCEPT_STATES, SIMPLIFIED_JSON_STATES, \
    SIMPLIFIED_JSON_ALPHABET
from community import modularity, best_partition
from networkx import dense_gnm_random_graph, Graph, simple_cycles
import numpy as np


def random_expected_modularity_frac(num_nodes, num_edges):
    gnx = dense_gnm_random_graph(num_nodes, num_edges).to_undirected()
    partition = best_partition(gnx)
    modularity(partition, gnx)
    in_partition = len([1 for u, v in gnx.edges() if partition[u] == partition[v]]) / len(gnx.edges)
    return in_partition - modularity(partition, gnx)


def simplified_json_more_edges(alphabet, states, init_state, accept_states, transitions, fst, add_edges_k):
    """
    adding K transitions without affecting
     - number of cycles
     - average degree
     - modularity score
    """
    # deg = 2 * len_edge / len_nodes
    # 2 * (len_edge + k) / (len_nodes + x) = deg
    # ( 2 * (len_edge + k) / deg ) - len_nodes = x
    # x = number of nodes to add with respect to k to preserve average degree
    # prove Ek > Nk

    alphabet = deepcopy(alphabet)
    states = deepcopy(states)
    fst = deepcopy(fst)
    accept_states = deepcopy(accept_states)
    gnx = deepcopy(fst.gnx)
    undirected_gnx = gnx.to_undirected()

    # get average_deg, number_of_cycles, number of nodes to add, number of accept/reject states to add
    average_deg = np.average(list(dict(undirected_gnx.degree(gnx.nodes)).values()))
    nodes_to_add = int(2 * (len(gnx.edges()) + add_edges_k) / average_deg - len(gnx.nodes()))
    reject_state_len = len([1 for state in fst.states() if state[1].is_reject])
    accept_nodes_to_add = int((len(gnx.nodes()) + nodes_to_add) * len(accept_states) / len(gnx.nodes())) \
                          - len(accept_states)
    reject_nodes_to_add = int((len(gnx.nodes()) + nodes_to_add) * reject_state_len / len(gnx.nodes())) \
                          - reject_state_len
    undirected_gnx = gnx.to_undirected()
    partition = best_partition(undirected_gnx)
    targe_modularity_score = modularity(partition, undirected_gnx)

    normal_states_to_add = ["state_" + str(i) for i in range(nodes_to_add - accept_nodes_to_add - reject_nodes_to_add)]
    accept_states_to_add = ["acc_state_" + str(i) for i in range(accept_nodes_to_add)]
    reject_states_to_add = ["rej_state_" + str(i) for i in range(reject_nodes_to_add)]


    # add extra states
    # {state : {edge: [sym] }
    new_transition_map = {}
    for tran in fst.full_transition_list():
        src, sym, dst = tran
        edges = new_transition_map.get(src, {})
        edges[dst] = edges.get(dst, []) + [sym]
        new_transition_map[src] = edges

    current_states = deepcopy(states)
    original_reject_states = [state[0] for state in fst.states() if state[1].is_reject]
    curr_reject_states = set(original_reject_states + reject_states_to_add)
    for state in accept_states_to_add + reject_states_to_add + normal_states_to_add:
        type_state = state.split("_")[0]
        # if accept or reject state pick one of current_state and add it
        if type_state != "state":
            source, sym, original_dst = get_source_and_sym(new_transition_map, current_states, curr_reject_states)

            new_transition_map[state] = {original_reject_states[0]: deepcopy(alphabet)}
            new_transition_map[source][original_dst].remove(sym)
            new_transition_map[source][state] = new_transition_map[source].get(state, []) + [sym]
            gnx.add_edge(source, state)

        # normal state add it between two existing states
        else:
            source, dest = choice(list(gnx.edges))
            gnx.remove_edge(source, dest)
            gnx.add_edges_from([(source, state), (state, dest)])
            tran_syms = new_transition_map[source][dest]
            for sym in tran_syms:
                new_transition_map[source][dest].remove(sym)  # remove all transition src - sym_1..n -> dst
                new_transition_map[source][state] = new_transition_map[source].get(state, []) + [sym]  # src - sym_1..n -> state
            new_transition_map[state] = {dest: deepcopy(alphabet)}                                # src - sym_1..n -> state - any -> dst

        current_states.add(state)

    # edges to add
    left_to_add = add_edges_k - accept_nodes_to_add - reject_nodes_to_add - len(normal_states_to_add)

    temp_gnx = deepcopy(gnx)
    undirected_gnx = gnx.to_undirected()
    partition = best_partition(undirected_gnx)
    communities = {}
    for state, community in partition.items():
        communities[community] = communities.get(community, []) + [state]

    in_partition_edges = len([1 for u, v in gnx.edges() if partition[u] == partition[v]])
    current_real = in_partition_edges / (len(gnx.edges) + left_to_add)

    expected = random_expected_modularity_frac(len(gnx.nodes()), len(gnx.edges) + left_to_add)

    target_real = targe_modularity_score + expected  # target_modularity = target_real - target_expected
    target_in_partition = round(target_real * (len(gnx.edges()) + left_to_add))
    to_add_in_partition = max(0, min(left_to_add, target_in_partition - in_partition_edges))
    to_add_between_partitions = left_to_add - to_add_in_partition

    print("in", to_add_in_partition, "out", to_add_between_partitions, "total", left_to_add)
    dest = None
    original_cycle = len(list(simple_cycles(gnx)))

    for i in range(to_add_between_partitions):
        source, sym, original_dst = get_source_and_sym(new_transition_map, current_states, curr_reject_states)
        source_community_complement = [k for k in communities if k != partition[source]]
        complement_states = []
        for comm in source_community_complement:
            complement_states += communities[comm]
        shuffle(complement_states)

        for d in complement_states:
            temp_gnx.add_edge(source, d)
            if len(list(simple_cycles(temp_gnx))) == original_cycle:
                dest = d
                break
            temp_gnx.remove_edge(source, d)
        # add between partition
        gnx.add_edge(source, dest)
        new_transition_map[source][original_dst].remove(sym)  # remove all transition src - sym_1..n -> dst
        new_transition_map[source][dest] = new_transition_map[source].get(dest, []) + [sym]


    # add rest of edges and preserve modularity
    dest = None
    for i in range(to_add_in_partition):
        source, sym, original_dst = get_source_and_sym(new_transition_map, current_states, curr_reject_states)
        in_comm_states = deepcopy(communities[partition[source]])
        in_comm_states.remove(source)
        if len(in_comm_states) > 0:
            shuffle(in_comm_states)
            for d in in_comm_states:
                temp_gnx.add_edge(source, d)
                if len(list(simple_cycles(temp_gnx))) == original_cycle:
                    dest = d
                    break
                temp_gnx.remove_edge(source, d)
        gnx.add_edge(source, dest)
        new_transition_map[source][original_dst].remove(sym)  # remove all transition src - sym_1..n -> dst
        new_transition_map[source][dest] = new_transition_map[source].get(dest, []) + [sym]

    undirected_gnx = gnx.to_undirected()
    print("cycles", original_cycle, len(list(simple_cycles(gnx))))
    print("deg", average_deg, np.average(list(dict(undirected_gnx.degree(gnx.nodes)).values())))
    partition = best_partition(undirected_gnx)
    modularity_score = modularity(partition, undirected_gnx)
    print("modularity", targe_modularity_score, modularity_score)
    e = 0

    new_alphabet = alphabet
    new_states = set(list(states) + accept_states_to_add + reject_states_to_add + normal_states_to_add)
    new_init_state = init_state
    new_accept_states = accept_states + accept_states_to_add
    new_transitions = []
    for source, edges in new_transition_map.items():
        for target, symbol_list in edges.items():
            for sym in symbol_list:
                new_transitions.append((source, sym, target))
    return (new_alphabet, new_states, new_init_state, new_accept_states, new_transitions),  \
            {"cycles": len(list(simple_cycles(gnx))),
             "deg": np.average(list(dict(undirected_gnx.degree(gnx.nodes)).values())),
             "modularity": modularity_score}


def get_source_and_sym(transition_map, states, reject_states):
    source = choice(list(states))
    sym, original_dst = None, None
    while source in reject_states or sym is None:
        source = choice(list(states))
        if source in reject_states:
            continue
        valid_symbols = []
        for dst, edge_sym in transition_map[source].items():
            if len(edge_sym) > 1:
                valid_symbols += [(sym, dst) for sym in edge_sym]
        if len(valid_symbols) <= 2:
            continue
        sym, original_dst = choice(valid_symbols)
    return source, sym, original_dst

# undirected_gnx = gnx.to_undirected()
# partition = best_partition(undirected_gnx)
# modularity_score = modularity(partition, undirected_gnx)  # M(gnx, partition) = Real - Expected
# expected = random_expected_modularity_frac(list(undirected_gnx.nodes()), len(SIMPLIFIED_JSON_TRANSITIONS), partition)
# real = modularity_score + expected


if __name__ == '__main__':
    to_add_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    all = {k: [] for k in to_add_list}
    for to_add in to_add_list:
        for repeat in range(10):
            tran, measures = simplified_json_more_edges(deepcopy(SIMPLIFIED_JSON_ALPHABET), deepcopy(SIMPLIFIED_JSON_STATES),
                                               deepcopy(SIMPLIFIED_JSON_INIT_STATE), deepcopy(SIMPLIFIED_JSON_ACCEPT_STATES),
                                               deepcopy(SIMPLIFIED_JSON_TRANSITIONS), deepcopy(SIMPLIFIED_JSON_FST), to_add)
            all[to_add].append((tran, measures))
    pickle.dump(all, open("all_tran.pkl", "wb"))



