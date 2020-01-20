from fst import FST, ACCEPT_SYMBOL
import itertools
from random import sample, choice
INTERSECT = "INTERSECT"
UNION = "UNION"


class FSTools:
    def __init__(self):
        pass

    def _merge_fst(self, list_fst, op=INTERSECT):
        # given list of fst [ Q, R, T ]
        # new states are [ ... (qi, rj, tk) ...]
        new_states = list(itertools.product(*(fst.state_list for fst in list_fst)))  # too expensive? n^(len_list)
        # new alphabet is the union of all alphabets
        new_alphabet = []
        for fst in list_fst:
            new_alphabet += fst.alphabet
        new_alphabet = list(set(new_alphabet))
        # new start state is (q_start, r_start, t_start) since we are limiting our FST to a single start state
        new_start_state = str(tuple(fst.start_state for fst in list_fst))
        # new accept state is dependent in operation
        # if intersection  -> accept states are [ ...(q_accept_i, r_accept_j, t_accept_k) ... ] all must be accept
        # if union         -> accept states are [ ...(q_accept_i, r_j, t_k) ... and so on )...] one must be accept
        new_accept_states = []
        new_transitions = []
        len_state = len(list_fst)
        # loop over states -> loop alphabet -> loop fst_list and fill transition dictionary
        # { ... (qi, rj, tk):{ ... symbol:(target, weight) ....} ... }
        for state in new_states:
            for symbol in new_alphabet:
                if symbol == ACCEPT_SYMBOL:         # this is an artificial symbol
                    continue
                # init target which is a tuple (q,r,k) in len of fst_list
                target = [0] * len_state
                # weigh will be the sum of all transitions
                weight = 0
                accept_weight = 0
                is_accept = False if op == UNION else True
                for i, (s, fst) in enumerate(zip(state, list_fst)):
                    # s is a single state name from tuple (q,r,t) -> after next line its a state object
                    s = fst.state(s)
                    # t-target, w-weight -> sum weights for the transition and calc target
                    t, w = s.transition(symbol)
                    weight += w
                    target[i] = t.id
                    # calculate accept weights
                    art_accept, w_accept = s.transition(ACCEPT_SYMBOL)
                    accept_weight += w_accept

                    # check if state is accept or not
                    if op == UNION and s.is_accept:
                        is_accept = True
                    if op == INTERSECT and not s.is_accept:
                        is_accept = False
                # add to accept states
                if is_accept:
                    new_accept_states.append((state, accept_weight))
                # add to transition dictionary
                new_transitions.append((state, symbol, tuple(target), weight))
        # convert all states to str
        new_states = set(str(s) for s in new_states)
        new_accept_states = list((str(s), w) for s, w in new_accept_states)
        new_transitions = [(str(src), symbol, str(dst), weight) for src, symbol, dst, weight in new_transitions]
        return new_alphabet, new_states, new_start_state, new_accept_states, new_transitions

    def intersection_fst(self, list_fst):
        new_alphabet, new_states, new_start_state, new_accept_states, new_transitions =\
            self._merge_fst(list_fst, op=INTERSECT)
        return FST(new_alphabet, new_states, new_start_state, new_accept_states, new_transitions)

    def union_fst(self, list_fst):
        new_alphabet, new_states, new_start_state, new_accept_states, new_transitions =\
            self._merge_fst(list_fst, op=UNION)
        return FST(new_alphabet, new_states, new_start_state, new_accept_states, new_transitions)

    def rand_fst(self, size_states, size_alphabet, num_accept_states):
        alphabet, states, start_state, accept_states, transitions = [], [], "q0", [], []
        for i in range(size_alphabet):
            alphabet.append("sym" + str(i))
        for i in range(size_states):
            states.append("q" + str(i))
        accept_states = [(q, 1) for q in sample(states, num_accept_states)]

        for q in states:
            for symbol in alphabet:
                transitions.append((q, symbol, choice(states)))
        return FST(alphabet, set(states), start_state, accept_states, transitions)
        # randomize according to: K1 states, K2 alphabet, fixed size / fixed per node ...
        # unique test and train
        # LSTM on a single automaton
        # negative samples generated from original FST not ending at accept state + Language model
        # pass


"""
 L1 all sequences that start and and with different letters
 a(a,b)*b || b(a,b)*a
 
      |   a   |   b   |
-----------------------
  s0  |  s1   |  s3   | 
  s1  |  s1   |  s2   |
  s2  |  s1   |  s2   | acc=1
  s3  |  s4   |  s3   |
  s4  |  s4   |  s3   | acc=1
  
  ===========================================================
  L2={ a^n b^n || n,m > 0 }
        |   a   |   b   |
-----------------------
  q0  |  q1   |  q3   |
  q1  |  q1   |  q2   |
  q2  |  q3   |  q2   | acc=3
  q3  |  q3   |  q3   |
"""

if __name__ == "__main__":
    #  L1
    _alphabet_L1 = ["a", "b"]
    _states_L1 = {"s0", "s1", "s2", "s3", "s4"}
    _init_state_L1 = "s0"
    _accept_states_L1 = [("s2", 1), ("s4", 1)]
    _transitions_L1 = [
        ("s0", "a", "s1"),
        ("s0", "b", "s3"),
        ("s1", "a", "s1"),
        ("s1", "b", "s2"),
        ("s2", "a", "s1"),
        ("s2", "b", "s2"),
        ("s3", "a", "s4"),
        ("s3", "b", "s3"),
        ("s4", "a", "s4"),
        ("s4", "b", "s3")
    ]
    _fst_L1 = FST(_alphabet_L1, _states_L1, _init_state_L1, _accept_states_L1, _transitions_L1)
    print("check FST - L1")
    print(_fst_L1)
    assert _fst_L1.go("aaabbababaabbab")[1]
    assert not _fst_L1.go("aaabbbbba")[1]
    rand = "".join(_fst_L1.go())
    print("sample:" + rand)
    assert _fst_L1.go(rand)

    #  L2
    _alphabet_L2 = ["a", "b"]
    _states_L2 = {"q0", "q1", "q2", "q3"}
    _init_state_L2 = "q0"
    _accept_states_L2 = [("q2", 3)]
    _transitions_L2 = [
        ("q0", "a", "q1"),
        ("q0", "b", "q3"),
        ("q1", "a", "q1"),
        ("q1", "b", "q2"),
        ("q2", "a", "q3"),
        ("q2", "b", "q2"),
        ("q3", "a", "q3"),
        ("q3", "b", "q3")
    ]
    _fst_L2 = FST(_alphabet_L2, _states_L2, _init_state_L2, _accept_states_L2, _transitions_L2)

    _union_fst = FSTools().union_fst([_fst_L1, _fst_L2])
    print("union - L1 or L2")
    print(_union_fst)
    assert _union_fst.go("aaabbababaabbab")[1]
    assert _union_fst.go("aaabbbbb")[1]
    assert not _union_fst.go("aaabbbbba")[1]
    for i in range(20):
        rand = "".join(_union_fst.go())
        print("sample:" + rand)
        assert _union_fst.go(rand)

    _intersection_fst = FSTools().intersection_fst([_fst_L1, _fst_L2])
    print("intersection - L1 and L2")
    print(_intersection_fst)
    assert not _intersection_fst.go("aaabbababaabbab")[1]
    assert _intersection_fst.go("aaabbbbb")[1]
    assert not _intersection_fst.go("aaabbbbba")[1]
    for i in range(20):
        rand = "".join(_intersection_fst.go())
        print("sample:" + rand)
        assert _intersection_fst.go(rand)
    _rand_fst = FSTools()._rand_fst(10, 5, 2)
    e = 0
