

"""
simplified json FST (without reject state)

                                      - , -> << q3 >> -> S N B L D -
                                     |                             \/
                   - S N B L D ->   <<             q2                >> -- ] -----
                  |                                                              \/
    -- [ ----> << q1 >> ------------------  ]  ---------------------------> <<   q4 accept >>
   |
< q0 start >
   |
    -- { ----> <<   q5   >> ------------------  }  -----------------------> <<   q9 accept >>
                  /\ |                                                                   /\
                  |   - S --> << q6 >> -- : --> << q7 >> -- S N B L D --> << q8 >> -- } --
                  |                                                       |
                   --------------------------- , -------------------------


      |   S   |   N   |   B   |   L   |   D   |   [   |   ]   |   {   |   }  |   ,   |   :   |
-----------------------------------------------------------------------------------------------
  q0  |  q10  |  q10  |  q10  |  q10  |  q10  |  q1   |  q10  |  q5   |  q10  |  q10  |  q10  |
  q1  |  q2   |  q2   |  q2   |  q2   |  q2   |  q10  |  q4   |  q10  |  q10  |  q10  |  q10  |
  q2  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q4   |  q10  |  q10  |  q3   |  q10  |
  q3  |  q2   |  q2   |  q2   |  q2   |  q2   |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |
  q4  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  | acc=1
  q5  |  q6   |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q9   |  q10  |  q10  |
  q6  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q7   |
  q7  |  q8   |  q8   |  q8   |  q8   |  q8   |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |
  q8  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q9   |  q5   |  q10  |
  q9  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  | acc=1
  q10 |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  |  q10  | rej=1

"""

from community import best_partition, modularity
from networkx import pagerank

from fst import FST
import numpy as np

SEQ_LEN = 100
SIMPLIFIED_JSON_ALPHABET = ["S", "N", "B", "L", "D", ":", ",", "[", "]", "{", "}"]
SIMPLIFIED_JSON_STATES = {"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"}
SIMPLIFIED_JSON_INIT_STATE = "q0"
SIMPLIFIED_JSON_ACCEPT_STATES = ["q4", "q9"]

SIMPLIFIED_JSON_TRANSITIONS = [
    ("q0", "[", "q1"),
    ("q0", "{", "q5"),

    ("q1", "S", "q2", 50),
    ("q1", "N", "q2", 50),
    ("q1", "B", "q2", 50),
    ("q1", "L", "q2", 50),
    ("q1", "D", "q2", 50),
    ("q1", "]", "q4", 1),

    ("q2", ",", "q3", SEQ_LEN),
    ("q2", "]", "q4", 1),

    ("q3", "S", "q2"),
    ("q3", "N", "q2"),
    ("q3", "B", "q2"),
    ("q3", "L", "q2"),
    ("q3", "D", "q2"),

    ("q5", "}", "q9", 1),
    ("q5", "S", "q6", 250),

    ("q6", ":", "q7"),

    ("q7", "S", "q8"),
    ("q7", "N", "q8"),
    ("q7", "B", "q8"),
    ("q7", "L", "q8"),
    ("q7", "D", "q8"),

    ("q8", ",", "q5", SEQ_LEN),
    ("q8", "}", "q9", 1)
]

SIMPLIFIED_JSON_FST = FST(SIMPLIFIED_JSON_ALPHABET, SIMPLIFIED_JSON_STATES, SIMPLIFIED_JSON_INIT_STATE,
                          SIMPLIFIED_JSON_ACCEPT_STATES, SIMPLIFIED_JSON_TRANSITIONS)
if __name__ == '__main__':
    _fst = SIMPLIFIED_JSON_FST
    print(_fst)
    # POSITIVES:
    assert _fst.go("{S:S}")[1]
    assert _fst.go("{S:N,S:S}")[1]
    assert _fst.go("{S:N,S:S,S:S,S:D,S:L,S:S,S:N,S:B,S:L,S:N,S:N,S:N,S:S,S:L,S:L}")[1]
    assert _fst.go("[B,L,D]")[1]
    assert _fst.go("[B,D,S,S,L]")[1]
    assert _fst.go("{S:L,S:L,S:D,S:B,S:N,S:L,S:L}")[1]
    assert _fst.go("[L,L,B,S,S,D,D,N,L,B,S,D,S,N]")[1]
    assert _fst.go("[L,B,B]")[1]
    assert _fst.go("{S:D,S:S,S:L,S:B,S:L,S:L,S:D,S:L,S:S,S:L,S:L,S:S,S:N,S:B,S:N}")[1]
    assert _fst.go("[]")[1]
    assert _fst.go("{S:B,S:L,S:B,S:N,S:S,S:L,S:B,S:D,S:B,S:S}")[1]
    # NEGATIVES:
    assert not _fst.go("{S{B,S:N}")[1]
    assert not _fst.go("{S:S,S:DNS:B,S:S,S:D}")[1]
    assert not _fst.go("{S:B,S:S,S:N,")[1]
    assert not _fst.go("{{:N,S:L,S:L,S:S,S:S,S:L,S:D,S:S}")[1]
    assert not _fst.go("{S:N,S:N,}:S,S:D,S:S}")[1]
    assert not _fst.go("[S,B,S,B,N,DLL,S,N,B,B,B,B]")[1]
    assert not _fst.go("{S:S,S:B,S:D,S:L,S:B,S:D,{:B,S:B,S:S,S:B}")[1]
    assert not _fst.go("[L,:,N,S,B,L,L,B,S,N,B,L,B]")[1]
    assert not _fst.go("{S:D,S:S,S:S,S:N,S:D,S:N,S:S,S:],S:L}")[1]
    assert not _fst.go("[D,N,S,S,LBD,D,N,N,N,N,L,B,D,D,N]")[1]

    for i in range(10):
        rand = "".join(_fst.go())
        print("sample " + str(i) + ":" + rand)
        assert _fst.go(rand)

    gnx = _fst.gnx
    undirected_gnx = gnx.to_undirected()
    partitions = [best_partition(undirected_gnx) for _ in range(10)]
    modularity_score = np.average([modularity(partition, undirected_gnx) for partition in partitions])
    page_rank_score = pagerank(gnx)

    stats = {
        "nodes / states": 10,
        "alphabet": 11,
        "cycle count": 2,
        "effective degree": np.average(list(_fst.effective_deg().values())),
        "modularity score": modularity_score,
        "page rank score": np.average(list(page_rank_score.values()))
    }
    for k, v in stats.items():
        print(k, v)
