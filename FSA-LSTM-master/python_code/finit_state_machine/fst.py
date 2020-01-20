from random import randint, choice
import networkx as nx
from random import randint
from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + r"D:\Program Files (x86)\Graphviz2.38\bin"
EPS_MOVE = "EPS_MOVE"
ACCEPT_SYMBOL = "ACC_SYM"
ART_ACCEPT_STATE = "ART_ACC_STATE"
REJECT_STATE = "REJ_STATE"
OUT_DEG = "_OUT_DEG_"
IN_DEG = "_IN_DEG_"
IN_AND_OUT_DEG = "_IN_OUT_DEG_"


class State:
    """
    this class represent a single state within an FST machine
    it includes its transitions ( can be weighted )
    init params
    state_name:     the states name ( which is the source )
    transition:     a dictionary of transitions { symbol: target_state, weight<int> }
                    ** if a transition (state, symbol) is not defined then its assumed to be reject **
    is_init:        True if its an initial state
    is_accept:      True if its an accept state
    """
    def __init__(self, state_name, reject_state, transitions: dict= None, is_init=False, is_accept=False,
                 is_reject=False, artificial_accept=False):
        self._source = state_name                               # states name
        self._reject_state = reject_state                       # reject state used in case invalid symbol is used
        self._is_initial_state = is_init
        self._is_accept_state = is_accept
        self._is_reject_state = is_reject
        self._is_art_accept_state = artificial_accept           # not real state from user input
        self._transition = transitions if transitions else {}   # transition dict { symbol: (target, weight) }
        self._weights = [("sym", "<<>>", 0)]
        self._edited = True                                     # true if a transition is edited
        self._base_weight = None                                # weight of last edited transition
        self._weighted = False                                  # true if there is different weight for transitions

    @property
    def id(self):
        return self._source

    @property
    def is_init(self):
        return self._is_initial_state

    @property
    def is_accept(self):
        return self._is_accept_state

    @property
    def is_art_accept(self):
        return self._is_art_accept_state

    @property
    def is_reject(self):
        return self._is_reject_state

    @property
    def source(self):
        return self._source

    @property
    def transitions(self):
        return self._transition

    def transition(self, symbol):
        return self._transition[symbol] if symbol in self._transition else (self._reject_state, 0)

    """
    this function edits a transition according to input 
    """
    def edit_transition(self, symbol, target, weight=1):
        self._edited = True
        # picking by weights is rather expensive, thus a flag will be raise if there are any weights at all
        if self._base_weight is None:
            self._base_weight = weight
        if weight != self._base_weight:
            self._weighted = True
        self._transition[symbol] = (target, weight)

    def _get_acceptable_weights(self):
        # if the machine was edited then update the weights
        if self._edited:
            # reset weights, weights is a list of tuples [ ... ( symbol, target, weight )
            self._weights = [("sym", "<<>>", 0)]
            # the weight of transition i is the difference between i and i-1
            i = 0
            for symbol, (target, weight) in self._transition.items():
                if not target.is_reject:
                    self._weights.append((symbol, target, self._weights[i][2] + weight))
                    i += 1
            # dismiss ("<<>>", 0)
            self._weights = (self._weights[1:], self._weights[-1][2])  # return weights + max weight
            self._edited = False
        return self._weights

    """
        this function randomly picks a transition, according to the weights of the model
    """
    def _rand_acceptable_with_weights(self):
        # get weights as intervals i.e. for symbols=weights <a=2,b=3,c=5>
        # weights is [2,5,10] and max weight is 10
        weights, max_weight = self._get_acceptable_weights()
        # rand a number between  0 <= n <= max - 1
        rand_num = randint(0, max_weight - 1)
        # loop over transitions, and return the first one that bigger then target,
        # if weights is empty return epsilon move
        for symbol, target, weight in weights:
            if weight > rand_num:
                return symbol, target
        return EPS_MOVE, self

    """
        this function randomly picks a transition, with no consideration to the weights of the model
    """
    def _rand_acceptable_without_weights(self):
        # if the transition list is empty return an epsilon move
        tran_list = [(symbol, target) for (symbol, (target, weight)) in self._transition.items()
                     if not target.is_reject]
        if not tran_list:
            return EPS_MOVE, self

        # else, randomly choose a transition
        symbol, target = choice(tran_list)
        return symbol, target

    """
    this function randomly picks a transition
    """
    def _rand_acceptable_transition(self):
        if self._weighted:
            return self._rand_acceptable_with_weights()
        return self._rand_acceptable_without_weights()

    """
    this function returns:
    if a symbol is given -> the next state is returned according the transition function 
    if no symbol is given -> a symbol is raffled according to the weights and then a transition=(symbol, next_state) 
    is returned, the next state cannot be a reject state in this case. 
    """
    def go(self, symbol=None):
        # if there's no transition rule registered for the symbol than state isn't changing
        if symbol is not None:
            if symbol not in self._transition:
                return self._reject_state
            return self._transition.get(symbol)[0]
        return self._rand_acceptable_transition()

    # end = True  -> generate a not accept state
    # end = False -> an accept state can be generate as long as it not utterly accept
    def go_negative(self):
        tran_list = [(symbol, target) for (symbol, (target, weight)) in self._transition.items()]
        if not tran_list:
            return EPS_MOVE, self

        # else, randomly choose a transition
        symbol, target = choice(tran_list)
        return symbol, target


class FST:
    """
    this class represent an FST machine
    init params:
    alphabet:       list of alphabet symbols
    states:         set<preferred>/list/tuple of states<strings/chars/int>
    start_state:    a single initial state
    accept_state:   list of accept states [ ...(qi, weight)...) or [ ...qi...] in that case weight are 1
    transitions:    list of tuples ( source_state, symbol, target_state, weight<int><optional>)
    """
    def __init__(self, alphabet, states: set, start_state, accept_states: list, transitions: list):
        self._alphabet = alphabet               # list [a, b, ...]
        self._states = states                   # list [q0, q1, ...]
        self._start_state = start_state         # q0
        # self._accept_states = [ ... (qi, weight) ... ]
        self._accept_states = [(s, 1) for s in accept_states] if type(accept_states[0]) is str else accept_states
        # dictionary { q: State(q) } ... State has an attribute->transition which is a dict { symbol : (State, weight) }
        # the dictionary includes the artificial ACCEPT_STATE { state_name: State object }
        self._transition_list = transitions
        self._transitions = self._build_transitions(states, start_state, self._accept_states, transitions)

    @property
    def state_list(self):
        return list(self._states)

    @property
    def state_names(self):
        return self._states

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def start_state(self):
        return self._start_state

    def accept_states(self, weight=False):
        for acc_state in self._accept_states:
            yield acc_state if weight else acc_state[0]

    def state(self, name):
        return self._transitions[name]

    def states(self):
        for name, state in self._transitions.items():
            if name == ART_ACCEPT_STATE:
                continue
            yield name, state

    def __str__(self):
        out_lines = []
        max_len_state = max(len(max(self._states, key=lambda x: len(x))) + 2, 20)
        max_len_alphabet = max(len(max(self._alphabet, key=lambda x: len(x))) + 2, 20)

        out_lines.append("Alphabet")
        for symbol in self._alphabet:
            out_lines.append(symbol)

        out_lines.append("\n\nStates")
        out_lines.append(" " * int((max_len_state - 6) / 2) + "Source" + " " * int((max_len_state - 6) / 2) + "||"
                         + "     type")

        for state in sorted(self._states):
            if state == ART_ACCEPT_STATE:
                continue
            out_lines.append(state + " " * int(max_len_state - len(state)) + "||"
                             + ("  -initial_state" if self._transitions[state].is_init else "")
                             + ("  -accept_state" if self._transitions[state].is_accept else "")
                             + ("  -reject_state" if self._transitions[state].is_reject else "")
                             )

        out_lines.append("\n\nTransitions")
        out_lines.append(" " * int((max_len_state - 6) / 2) + "Source" + " " * int((max_len_state - 6) / 2) + "||" +
                         " " * int((max_len_alphabet - 6) / 2) + "Symbol" + " " * int((max_len_alphabet - 6) / 2) + "||" +
                         " " * int((max_len_state - 6) / 2) + "Target" + " " * int((max_len_state - 6) / 2) + "||" +
                         "     Weight"
                         )

        for state in sorted(self._states):
            if state == ART_ACCEPT_STATE:
                continue
            tran = self._transitions[state].transitions
            for symbol, (target, weight) in tran.items():
                out_lines.append(state + " " * int(max_len_state - len(state)) + "||"
                                 + " " + symbol + " " * int((max_len_alphabet - len(symbol) - 1)) + "||"
                                 + " " + target.id + " " * int((max_len_state - len(target.id) - 1)) + "||"
                                 + " " + str(weight)
                                 )
        return "\n".join(out_lines)

    """
    this function gets for an input a full transition function for the FST and an accept state
    the function return the reject states for the FST
    
     - a graph (gnx) is generated according to the transition 
     - check if path exists from node to accept_state for all nodes in V -> if there isn't then it's a reject_state
    """
    def _get_reject_states(self, transitions, accept_states):
        # build FST graph
        gnx = nx.DiGraph()
        list_edges = []
        for tran in transitions:
            # discard symbols and weights
            source, symbol, target, weight = tran if len(tran) == 4 else list(tran) + [1]
            list_edges.append((source, target))
        gnx.add_edges_from(list_edges)
        # check if there is a path from every node(state) to accept state if not add to reject_states
        reject_states = set()
        for node in gnx:
            reject_flag = True
            for acc_state, acc_weight in accept_states:
                # if there is a path to at least one accept node
                if nx.has_path(gnx, node, acc_state):
                    reject_flag = False
            if reject_flag:
                reject_states.add(node)
        return reject_states

    """
    this function builds a transition dictionary:
    { state_name: State_object }
    the state object includes all transitions and weight for a specific state - more info above at State class   
    """
    def _build_transitions(self, states, init_state, accept_states, transitions):
        accept_set = set([s for s, w in accept_states])
        reject_states = self._get_reject_states(transitions, accept_states)
        global_reject_state = State(REJECT_STATE, None, is_reject=True)  # artificial reject state
        global_reject_state._reject_state = global_reject_state
        # build a dictionary of name to State objects
        state_dict = {q: State(q, global_reject_state, is_init=q == init_state, is_accept=q in accept_set,
                               is_reject=q in reject_states) for q in self._states}
        self._states.add(REJECT_STATE)
        state_dict[REJECT_STATE] = global_reject_state
        # add a artificial node for accept state
        state_dict[ART_ACCEPT_STATE] = State(ART_ACCEPT_STATE, global_reject_state, is_accept=True, artificial_accept=True)
        for acc_state, acc_weight in accept_states:
            transitions.append((acc_state, ACCEPT_SYMBOL, ART_ACCEPT_STATE, acc_weight))
        for tran in transitions:
            source, symbol, target, weight = tran if len(tran) == 4 else list(tran) + [1]
            state_dict[source].edit_transition(symbol, state_dict[target], weight=weight)
        return state_dict

    """
    this function runs the machine 
    if a sequence is given -> the function returns the final state and if its accepted by the  machine or not
    if no sequence is given -> the function randomly shuffles according to the weights and returns the sequence
    """
    def go(self, sequence=None):
        # start at initial state
        curr_state = self._transitions[self._start_state]
        # activate states sequentially and return final state
        if sequence is not None:
            for symbol in sequence:
                curr_state = self._transitions[curr_state.id].go(symbol)
                if curr_state.is_reject:
                    break
            return curr_state, curr_state.is_accept
        else:
            # start from an empty sequence
            sequence = []
            if curr_state.is_reject:
                return []
            # shuffle symbols until accepted
            while not curr_state.is_art_accept:
                symbol, curr_state = self._transitions[curr_state.id].go()
                sequence.append(symbol)
            return sequence[:-1]

    # TODO think of stopping condition// maybe save the gnx?
    def generate_negative(self, sample_len=randint(1, 50)):

        negative_sample_generated = False
        # shuffle symbols until negative sample was generated
        while not negative_sample_generated:
            # start at initial state
            curr_state = self._transitions[self._start_state]
            # start from an empty sequence
            sequence = []

            for _ in range(sample_len):
                symbol, curr_state = self._transitions[curr_state.id].go_negative()
                if symbol == EPS_MOVE:
                    break
                sequence.append(symbol)

            # check if the sample is negative
            if not curr_state.is_accept:
                negative_sample_generated = True
        return sequence

    def generate_relative_negative(self, list_accept: set, accepted_seq_only=False, sample_len=randint(1, 50)):
        negative_sample_generated = False
        # shuffle symbols until negative sample was generated
        while not negative_sample_generated:
            # start at initial state
            curr_state = self._transitions[self._start_state]
            # start from an empty sequence
            sequence = []
            for idx in range(sample_len):
                symbol, curr_state = self._transitions[curr_state.id].go_negative()
                if symbol == EPS_MOVE or curr_state.is_art_accept:
                    break
                sequence.append(symbol)

            # check if the sample is negative
            if not accepted_seq_only and not curr_state.is_accept:                # negative reject state sample
                negative_sample_generated = True
            elif curr_state.is_accept and curr_state.id not in list_accept:       # negative accept state sample
                negative_sample_generated = True

        return sequence

    def effective_deg(self, deg_type=IN_AND_OUT_DEG):
        # build FST graph
        gnx = self.gnx
        if deg_type == IN_AND_OUT_DEG:
            deg = dict(gnx.degree(gnx.nodes))
        if deg_type == IN_DEG:
            deg = dict(gnx.in_degree(gnx.nodes))
        if deg_type == OUT_DEG:
            deg = dict(gnx.out_degree(gnx.nodes))
        return deg

    @property
    def gnx(self):
        gnx = nx.DiGraph()
        list_edges = []
        for state_id in self._states:
            source_state = self._transitions[state_id]
            if source_state.is_art_accept or source_state.is_reject:
                continue

            for sym in self._alphabet:
                target_state = source_state.go(sym)
                if target_state.is_art_accept:
                    continue
                list_edges.append((source_state.id, target_state.id))

        gnx.add_edges_from(list_edges)
        return gnx

    def full_transition_list(self):
        full_transitions = []
        for state_id in self._states:
            source_state = self._transitions[state_id]
            if source_state.is_art_accept or source_state.is_reject:
                continue
            for sym in self._alphabet:
                target_state = source_state.go(sym)
                if target_state.is_art_accept:
                    continue
                full_transitions.append((source_state.id, sym, target_state.id))
        return full_transitions

    def mean_variance_sequence_len(self, num_samples=100):
        import numpy as np
        sequence_len = []
        for _ in range(num_samples):
            sequence_len.append(len(self.go()))
        return np.mean(sequence_len), np.std(sequence_len)

    def accept_percentage(self, min_len=1, max_len=100, num_samples=100):
        accepted = 0
        for sample_num in range(num_samples):
            sample_len = randint(min_len, max_len)

            # start at initial state
            curr_state = self._transitions[self._start_state]
            for _ in range(sample_len):
                symbol, curr_state = self._transitions[curr_state.id].go_negative()

            # check if the sample is negative
            if curr_state.is_accept:
                accepted += 1
        return accepted / num_samples

    def plot_fst(self, name="fst", path=os.path.join("graphviz_fst")):
        dot = Digraph(comment=name, format="svg")
        # add states as nodes to the graph + label is_accept/is_reject/is_init
        for state in sorted(self._states):
            if state == ART_ACCEPT_STATE or state == REJECT_STATE:
                continue
            dot.node(state,
                     state + " "
                     + ("  -initial_state" if self._transitions[state].is_init else "")
                     + ("  -accept_state" if self._transitions[state].is_accept else "")
                     + ("  -reject_state" if self._transitions[state].is_reject else "")
                     )
        # add transition as edges to the graph
        for tran in self._transition_list:
            # discard symbols and weights
            source, symbol, target, weight = tran if len(tran) == 4 else list(tran) + [1]
            if source == ART_ACCEPT_STATE or target == ART_ACCEPT_STATE:
                continue
            dot.edge(source, target, label=symbol)

        dot.render(path, view=False)


"""
test: the following FST represents the language L={ a^n b^n || n,m > 0 }

                             - a -
                            |     |
                            |     \/
<< q0 start >> --- a --->  << q1 >>
   |                          |
   b                          b
   \/                         \/
<< q3 >> <------- a -----  << q2 >>
|     /\                   |     /\             
|     |                    |     |                   
 - a,b                      - b -               
 
 verse-1 without-weights
      |   a   |   b   |
-----------------------
  q0  |  q1   |  q3   |
  q1  |  q1   |  q2   |
  q2  |  q3   |  q2   | acc=3
  q3  |  q3   |  q3   |
  
  
 verse-2 with weights -> weights are such that 'b' sould be longer 
      |   a=weight   |   b=weight   |  acc=weight
-------------------------------------------------
  q0  |     q1=1     |     q3=1     |
  q1  |     q1=3     |     q2=7     |
  q2  |     q3=1     |     q2=7     |  acc=3
  q3  |     q3=1     |     q3=1     |
"""

if __name__ == "__main__":

    # Verse-1
    _alphabet = ["a", "b"]
    _states = {"q0", "q1", "q2", "q3"}
    _init_state = "q0"
    _accept_states = [("q2", 3)]
    _transitions = [
        ("q0", "a", "q1"),
        ("q0", "b", "q3"),
        ("q1", "a", "q1"),
        ("q1", "b", "q2"),
        ("q2", "a", "q3"),
        ("q2", "b", "q2"),
        ("q3", "a", "q3"),
        ("q3", "b", "q3")
    ]
    _fst = FST(_alphabet, _states, _init_state, _accept_states, _transitions)
    print("unweighted FST")
    print(_fst)
    assert _fst.go("aaabbbbb")[1]
    assert not _fst.go("aaabbbbba")[1]
    rand = "".join(_fst.go())
    print("sample:" + rand)
    assert _fst.go(rand)

    # Verse-2
    _alphabet = ["a", "b"]
    _states = {"q0", "q1", "q2", "q3"}
    _init_state = "q0"
    _accept_states = [("q2", 3)]
    _transitions = [
        ("q0", "a", "q1"),
        ("q0", "b", "q3"),
        ("q1", "a", "q1", 3),
        ("q1", "b", "q2", 7),
        ("q2", "a", "q3"),
        ("q2", "b", "q2", 7),
        ("q3", "a", "q3"),
        ("q3", "b", "q3")
    ]
    _fst = FST(_alphabet, _states, _init_state, _accept_states, _transitions)
    print("\n\nweighted FST")
    print(_fst)
    assert _fst.go("aaabbbbb")[1]
    assert not _fst.go("aaabbbbba")[1]
    rand = "".join(_fst.go())
    print("sample:" + rand)
    assert _fst.go(rand)
    for i in range(100):
        _neg = _fst.generate_negative(sample_len=i*2 + 3)
        print("".join(_neg))
        assert not _fst.go(_neg)[1]
    e = 0
