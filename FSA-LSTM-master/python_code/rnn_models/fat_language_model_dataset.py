from copy import deepcopy

from torch import Tensor
from torch.utils.data import Dataset
from random import shuffle
import numpy as np
from fst_tools import FSTools
from dataset_params import FSTParams

PAD = "_PAD_"
END = "_END_"
START = "_START_"


class FstLanguageModuleDataset(Dataset):
    def __init__(self, parmas: FSTParams, fst=None, ready=None):
        """
        ready sould be a dictionary { fst_object, char_embeddings_dictionary, the data itself }
        fst is an fst_object
        only one should be used
        """
        if ready is not None:
            self._build_ready(ready)
            return
        self._fst = FSTools().rand_fst(parmas.FST_STATES_SIZE, parmas.FST_ALPHABET_SIZE, num_accept_states=2) \
            if fst is None else fst
        self._chr_embed, self._label_to_idx = self._get_embeddings(self._fst.alphabet)  # index alphabet for embeddings
        self._idx_to_chr = [c for c, idx in sorted(self._chr_embed.items(), key=lambda x: x[1])]
        self._data = self._build_data(parmas.DATASET_SIZE)  # get data

    def idx_to_char(self, idx):
        return self._idx_to_chr[idx]

    @property
    def fst(self):
        return self._fst

    @property
    def chr_embed(self):
        return self._chr_embed

    @property
    def label_to_idx_dict(self):
        return self._label_to_idx

    def _build_ready(self, ready_data):
        self._fst = ready_data["fst"]
        self._chr_embed = ready_data["chr_embed"]
        self._data = ready_data["data"]
        self._label_to_idx = ready_data["label_to_idx"]
        self._idx_to_chr = [c for c, idx in sorted(self._chr_embed.items(), key=lambda x: x[1])]

    def _get_embeddings(self, alphabet):
        embed = {symbol: i for i, symbol in enumerate(alphabet)}
        embed[END] = len(embed)                                     # special index for Start sequence
        embed[PAD] = len(embed)                                     # special index for padding
        label_to_idx = deepcopy(embed)
        embed[START] = len(embed)                                   # special index for start
        return embed, label_to_idx

    @property
    def pad_idx(self):
        return self._label_to_idx[PAD]                         # pad gets the last index

    @property
    def end_idx(self):
        return self._label_to_idx[END]                         # pad gets the last index

    def _build_data(self, size):
        # add positive samples (sample , 1)
        data, temp, data_set = [], [], set()
        while len(temp) < size:
            # generate examples of type one
            seq = self._fst.go()
            if str(seq) not in data_set:
                data_set.add(str(seq))
                temp.append([self._chr_embed[symbol] for symbol in seq])

        for embed in list(set(tuple(x) for x in temp))[:size]:
            data.append(([self._chr_embed[START]] + list(embed), list(embed) + [self._label_to_idx[END]]))
            # example of sample :   sequence:             fruit    flies    like       a        banana
            #                       x       : E(START)   E(fruit) E(flies) E(like)    E(a)     E(banana) .. E(PAD)
            #                       label   : E(fruit)   E(flies)  E(like)  E(a)    E(banana)   E(END)      E(PAD)
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

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def split_fst_language_model_dataset(dataset: FstLanguageModuleDataset, split_list):
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
            "fst": dataset.fst,
            "chr_embed": dataset.chr_embed,
            "data": new_data[i],
            "label_to_idx": dataset.label_to_idx_dict
        }
        sub_datasets.append(FstLanguageModuleDataset(None, ready=ready_dict))
    return sub_datasets


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = FstLanguageModuleDataset(FSTParams())
    dl = DataLoader(
        dataset=ds,
        batch_size=2,
        collate_fn=ds.collate_fn
    )

    for i_, (sequence_, label_) in enumerate(dl):
        # print(i_, sequence_, label_)
        for sample_seq, sample_label in zip(sequence_.tolist(), label_.tolist()):
            for i, j in zip(sample_seq, sample_label):
                print(i, j)
    e = 0
