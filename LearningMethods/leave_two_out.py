import itertools


class LeaveTwoOut:
    def __init__(self):
        pass

    def split(self, y):
        idx = list(range(len(y)))
        y_pos_idx = [i for i, y in enumerate(y) if y == 1]
        y_neg_idx = [i for i, y in enumerate(y) if y == 0]
        test = [[a, b] for a, b in list(itertools.product(y_pos_idx, y_neg_idx))]
        train = []
        for t in test:
            index_list = idx.copy()
            index_list.pop(max(t[0], t[1]))
            index_list.pop(min(t[0], t[1]))
            train.append(index_list)

        combinations = [(tr, te) for tr, te in zip(train, test)]
        return combinations