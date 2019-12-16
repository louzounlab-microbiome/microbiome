def merge_paths(t1, t2):
    merged_list = []
    if type(t1) == list:
        for _1 in t1:
            merged_list.append(_1)
    elif t1:
        merged_list.append(t1)
    if type(t2) == list:
        for _2 in t2:
            merged_list.append(_2)
    elif t2:
        merged_list.append(t2)
    return merged_list


def up_and_right(i, j, n, k, track=""):
    # ----- stopping condition -----
    if [i, j] == [n, k]:  # got to the square
        return track
    elif i > n or j > k:  # passed the square
        return None

    # ----- step -----
    # recursively compute all the paths that continue from this point- i, j
    t1 = up_and_right(i+1, j, n, k, track + "u")  # try to move up
    t2 = up_and_right(i, j+1, n, k, track + "r")  # try to move right

    if t1 and t2:  # both steps found a successful track
        return merge_paths(t1, t2)  # combine results into one list
    elif t1:
        return t1  # only up step found a successful track
    elif t2:
        return t2  # only right step found a successful track

"""
def up_and_right_2(n, k, lst):
    if len(lst) == 0:  # start the track string
        lst = ["" for track in range(n + k)]
    # ----- stopping condition -----
    if len(lst[]) == n + k:  # got to the square
        lst.append(track)

    # ----- step -----
    # recursively compute all the paths that continue from this point- i, j
    up_and_right_2(n, k, track + "u")  # try to move up
    up_and_right_2(n, k, track + "r")  # try to move right
"""


if __name__ == "__main__":
    paths = up_and_right(0, 0, 5, 3)
    # paths = up_and_right_2(0, 0, 5, 3)
    print(paths)


