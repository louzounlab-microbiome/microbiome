from random import shuffle
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from numpy.random import uniform, choice, randint
import ast
KEY_SYMBOL = 'S'
VAL_SYMBOLS = ['S', 'N', 'B', 'L', 'D']
DICTIONARY, ARRAY = 0, 1
ALL_SYMBOLS = ['S', 'N', 'B', 'L', 'D', "[", "]", "{", "}", ",", ":"]

PAD = "_PAD_"
END = "_END_"
START = "_START_"
EMBEDDINGS = ['S', 'N', 'B', 'L', 'D', "[", "]", "{", "}", ",", ":", END, PAD, START]
LABELS = ['S', 'N', 'B', 'L', 'D', "[", "]", "{", "}", ",", ":", END, PAD]


class SimplifiedJsonSequence:
    def __init__(self):
        pass

    @staticmethod
    def pos_sequence(sample_len):
        casing = randint(2)
        if casing is DICTIONARY:
            sequence = "{"
            for i in range(sample_len):
                sequence += KEY_SYMBOL + ":" + choice(VAL_SYMBOLS)
                if i != sample_len - 1:
                    sequence += ","
            sequence += "}"
            return sequence

        if casing is ARRAY:
            return str([choice(VAL_SYMBOLS) for _ in range(sample_len)]).replace("\'", "").replace(" ", "")

    @staticmethod
    def is_valid(sequence: str):
        if len(sequence) == 0:
            return True
        # ARRAY
        if sequence[0] == "[" and sequence[-1] == "]":
            try:
                seq = sequence.strip("[]").split(",")
                for symbol in seq:
                    if symbol not in VAL_SYMBOLS:
                        return False
            except:
                return False
            return True

        # DICTIONARY
        if sequence[0] == "{" and sequence[-1] == "}":
            seq = sequence.strip("{}").split(",")
            for item in seq:
                try:
                    key, val = item.split(":")
                    if key != KEY_SYMBOL or val not in VAL_SYMBOLS:
                        return False
                except:
                    return False
            return True
        return False

    @staticmethod
    def neg_sequence(sample_len):
        sj = SimplifiedJsonSequence()
        sequence = sj.pos_sequence(sample_len)
        # prefer to change uup to tenth of sequence symbols
        req_change = int(sample_len/10)
        min_changes = randint(1, req_change if req_change > 1 else 2)

        sequence = list(sequence)
        for _ in range(min_changes):
            idx = randint(0, len(sequence))
            sequence[idx] = choice(ALL_SYMBOLS)

        # case still valid keep changing
        while sj.is_valid("".join(sequence)):
            idx = randint(0, len(sequence))
            sequence[idx] = choice(ALL_SYMBOLS)
        return "".join(sequence)


class SimpleJsonAcceptorDataset(Dataset):
    def __init__(self, size=10000, ready=None):
        self._size = size
        if ready is not None:
            self._build_ready(ready)
            return
        self._idx_to_chr, self._chr_embed = EMBEDDINGS, {sym: idx for idx, sym in enumerate(EMBEDDINGS)}
        self._data = self._build()

    @property
    def chr_embed(self):
        return self._chr_embed

    def _build_ready(self, ready_data):
        self._idx_to_chr = ready_data["_idx_to_chr"]
        self._chr_embed = ready_data["_chr_embed"]
        self._data = ready_data["data"]

    def _build(self):
        sj = SimplifiedJsonSequence()
        exist = set()
        data = []
        negative_size = int(self._size / 2)
        positive_size = self._size - negative_size
        # build negative
        samples_len = [abs(int(x)) for x in uniform(0, 100, negative_size)]
        for sample_len in samples_len:
            created = False
            while not created:
                sequence = sj.neg_sequence(sample_len)

                if sequence not in exist:
                    created = True
                    sample = [self._chr_embed[sym] for sym in sequence] + [self._chr_embed[END]]
                    data.append((sample, 0))
                    exist.add(sequence)
                sample_len = randint(100)

        # Build positive
        samples_len = [abs(int(x)) for x in uniform(0, 100, positive_size)]
        for sample_len in samples_len:
            created = False
            while not created:
                sequence = sj.pos_sequence(sample_len)

                if sequence not in exist:
                    created = True
                    sample = [self._chr_embed[sym] for sym in sequence] + [self._chr_embed[END]]
                    data.append((sample, 1))
                    exist.add(sequence)
                sample_len = randint(100)
        shuffle(data)
        return data

    # function for torch Dataloader - creates batch matrices using Padding
    def collate_fn(self, batch):
        lengths_sequences = []
        # calculate max word len + max char len
        for sample, label in batch:
            lengths_sequences.append(len(sample))

        # in order to pad all batch to a single dimension max length is needed
        max_lengths_sequences = np.max(lengths_sequences)

        # new batch variables
        lengths_sequences_batch = []
        labels_batch = []
        for sample, label in batch:
            # pad word vectors
            lengths_sequences_batch.append([self._chr_embed[PAD]] * (max_lengths_sequences - len(sample)) + sample)
            labels_batch.append(label)

        return Tensor(lengths_sequences_batch).long(), Tensor(labels_batch).long()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]


class SimpleJsonLanguageModelDataset(Dataset):
    def __init__(self, size=10000, ready=None):
        self._size = size
        if ready is not None:
            self._build_ready(ready)
            return
        self._labels, self._label_to_idx = LABELS, {sym: idx for idx, sym in enumerate(LABELS)}
        self._idx_to_chr, self._chr_embed = EMBEDDINGS, {sym: idx for idx, sym in enumerate(EMBEDDINGS)}
        self._data = self._build()  # get data

    @property
    def pad_idx(self):
        return self._label_to_idx[PAD]

    @property
    def end_idx(self):
        return self._label_to_idx[END]

    def _build_ready(self, ready_data):
        self._labels = ready_data["_labels"]
        self._label_to_idx = ready_data["_label_to_idx"]
        self._idx_to_chr = ready_data["_idx_to_chr"]
        self._chr_embed = ready_data["_chr_embed"]
        self._data = ready_data["data"]

    def _build(self):
        sj = SimplifiedJsonSequence()
        exist = set()
        data = []
        negative_size = int(self._size / 2)
        positive_size = self._size - negative_size
        # build negative
        samples_len = [abs(int(x)) for x in uniform(0, 100, negative_size)]
        for sample_len in samples_len:
            created = False
            while not created:
                sequence = sj.neg_sequence(sample_len)

                if sequence not in exist:
                    created = True
                    exist.add(sequence)
                    sample = [self._chr_embed[START]] + [self._chr_embed[sym] for sym in sequence]
                    label = sample[1:] + [self._label_to_idx[END]]
                    data.append((sample, label))
                sample_len = randint(100)
        # Build positive
        samples_len = [abs(int(x)) for x in uniform(0, 100, positive_size)]
        for sample_len in samples_len:
            created = False
            while not created:
                sequence = sj.pos_sequence(sample_len)

                if sequence not in exist:
                    created = True
                    exist.add(sequence)
                    sample = [self._chr_embed[START]] + [self._chr_embed[sym] for sym in sequence]
                    label = sample[1:] + [self._label_to_idx[END]]
                    data.append((sample, label))
                sample_len = randint(100)
        shuffle(data)
        return data

    # function for torch Dataloader - creates batch matrices using Padding
    def collate_fn(self, batch):
        lengths_sequences = []
        # calculate max word len + max char len
        for sample, label in batch:
            lengths_sequences.append(len(sample))

        # in order to pad all batch to a single dimension max length is needed
        max_lengths_sequences = np.max(lengths_sequences)

        # new batch variables
        lengths_sequences_batch = []
        labels_batch = []
        for sample, label in batch:
            # pad word vectors
            lengths_sequences_batch.append(sample + [self._chr_embed[PAD]] * (max_lengths_sequences - len(sample)))
            labels_batch.append(label + [self._label_to_idx[PAD]] * (max_lengths_sequences - len(sample)))
        return Tensor(lengths_sequences_batch).long(), Tensor(labels_batch).long()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]


def split_simple_json_language_model_dataset(dataset: SimpleJsonLanguageModelDataset, split_list):
    """
    this function splits a data-set into n = len(split_list) disjointed data-sets
    """
    import numpy as np
    # create a list of lengths [0.1, 0.4, 0.5] -> [100, 500, 1000(=len_data)]
    split_list = np.multiply(np.cumsum(split_list), len(dataset)).astype("int").tolist()
    # list of shuffled indices to sample randomly
    shuffled_idx = list(range(len(dataset)))
    shuffle(shuffled_idx)
    # split the data itself
    new_data = [[] for _ in range(len(split_list))]
    for sub_data_idx, (start, end) in enumerate(zip([0] + split_list[:-1], split_list)):
        for i in range(start, end):
            new_data[sub_data_idx].append(dataset.__getitem__(shuffled_idx[i]))
    # create sub sets
    sub_datasets = []
    for i in range(len(new_data)):
        ready_dict = {
            "_labels": dataset._labels,
            "_label_to_idx": dataset._label_to_idx,
            "_chr_embed": dataset._chr_embed,
            "_idx_to_chr": dataset._idx_to_chr,
            "data": new_data[i],
        }
        sub_datasets.append(SimpleJsonLanguageModelDataset(dataset._size, ready=ready_dict))
    return sub_datasets


def split_simplified_json_acceptor_dataset(dataset: SimpleJsonAcceptorDataset, split_list):
    """
    this function splits a data-set into n = len(split_list) disjointed data-sets
    """
    import numpy as np
    # create a list of lengths [0.1, 0.4, 0.5] -> [100, 500, 1000(=len_data)]
    split_list = np.multiply(np.cumsum(split_list), len(dataset)).astype("int").tolist()
    # list of shuffled indices to sample randomly
    shuffled_idx = list(range(len(dataset)))
    shuffle(shuffled_idx)
    # split the data itself
    new_data = [[] for _ in range(len(split_list))]
    for sub_data_idx, (start, end) in enumerate(zip([0] + split_list[:-1], split_list)):
        for i in range(start, end):
            new_data[sub_data_idx].append(dataset.__getitem__(shuffled_idx[i]))
    # create sub sets
    sub_datasets = []
    for i in range(len(new_data)):
        ready_dict = {
            "_idx_to_chr": dataset._idx_to_chr,
            "_chr_embed": dataset._chr_embed,
            "data": new_data[i]
        }
        sub_datasets.append(SimpleJsonAcceptorDataset(dataset._size, ready=ready_dict))
    return sub_datasets


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # binary_ds_ = SimpleJsonAcceptorDataset()
    # a_, b_, c_ = split_simplified_json_acceptor_dataset(binary_ds_, [0.1, 0.4, 0.5])
    #
    # binary_dl_ = DataLoader(
    #     a_,
    #     batch_size=10,
    #     shuffle=True,
    #     collate_fn=binary_ds_.collate_fn
    # )
    #
    # for i, (sample, label) in enumerate(binary_dl_):
    #     print(sample, label)

    language_ds_ = SimpleJsonLanguageModelDataset()
    a_, b_, c_ = split_simple_json_language_model_dataset(language_ds_, [0.1, 0.4, 0.5])

    language_dl_ = DataLoader(
        a_,
        batch_size=10,
        shuffle=True,
        collate_fn=language_ds_.collate_fn
    )

    for i, (sample, label) in enumerate(language_dl_):
        print(sample, label)
