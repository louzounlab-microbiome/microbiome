from torch import Tensor
from torch.utils.data import Dataset
from random import shuffle
import numpy as np
from fst_tools import FSTools
from dataset_params import FSTParams

PAD = "_PAD_"
END = "_END_"


class FstDataset(Dataset):
    def __init__(self, params: FSTParams, fst=None, ready=None):
        """
        ready sould be a dictionary { fst_object, char_embeddings_dictionary, the data itself }
        fst is an fst_object
        only one should be used
        """
        if ready is not None:
            self._build_ready(ready)
            return
        self._fst = self._load_fst(params, fst)
        self._chr_embed = self._get_embeddings(self._fst.alphabet)                   # index alphabet for embeddings
        self._data = self._build_data(params.DATASET_SIZE, params.NEGATIVE_SAMPLES)  # get data

    @property
    def fst(self):
        return self._fst

    @property
    def chr_embed(self):
        return self._chr_embed

    def _build_ready(self, ready_data):
        self._fst = ready_data["fst"]
        self._chr_embed = ready_data["chr_embed"]
        self._data = ready_data["data"]

    def _load_fst(self, params: FSTParams, fst_to_load):
        if fst_to_load is not None:
            return fst_to_load
        fst = FSTools().rand_fst(params.FST_STATES_SIZE, params.FST_ALPHABET_SIZE, params.FST_ACCEPT_STATES_SIZE)
        while fst.state("q0").is_reject:
            fst = FSTools().rand_fst(params.FST_STATES_SIZE, params.FST_ALPHABET_SIZE, params.FST_ACCEPT_STATES_SIZE)
        return fst

    def _get_embeddings(self, alphabet):
        embed = {symbol: i for i, symbol in enumerate(alphabet)}
        embed[END] = len(embed)                                     # special index for Start sequence
        embed[PAD] = len(embed)                                     # special index for padding
        return embed

    def _build_data(self, size, negative):
        negative_size = size // 2 if negative else 0
        positive_size = size - negative_size
        positive_set, negative_set = set(), set()
        positive, negative = [], []
        # add positive samples

        while len(positive) < positive_size:
            # data.append(([self._chr_embed[symbol] for symbol in self._fst.go()], 1))
            seq = self._fst.go()
            if str(seq) not in positive_set:
                positive_set.add(str(seq))
                positive.append([self._chr_embed[symbol] for symbol in seq])
        # add negative samples (sample , acc_state_type)
        positive = [(list(x) + [self._chr_embed[END]], 1) for x in set(tuple(x) for x in positive)]

        i = 0
        while len(negative) < negative_size:
            # data.append(([self._chr_embed[symbol] for symbol in
            #               self._fst.generate_negative(max_size=len(data[i][0]) + 1)], 0))
            seq = self._fst.generate_negative(sample_len=len(positive[i % len(positive)][0]) + 1)
            if str(seq) not in negative_set:
                negative_set.add(str(seq))
                negative.append([self._chr_embed[symbol] for symbol in seq])
                i += 1
        negative = [(list(x) + [self._chr_embed[END]], 0) for x in set(tuple(x) for x in negative)]

        data = negative + positive
        # shuffle(data)
        # data = list(sorted(data, key=lambda x: len(x[0])))
        chunks = {}
        for seq, label in data:
            chunks[len(seq)] = chunks.get(len(seq), []) + [(seq, label)]
        shuffled_data = []
        for len_sample, chunk in sorted(chunks.items(), key=lambda x: x[0]):
            shuffle(chunk)
            shuffled_data += chunk
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

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def split_fst_dataset(dataset: FstDataset, split_list):
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
            "data": new_data[i]
        }
        sub_datasets.append(FstDataset(None, ready=ready_dict))
    return sub_datasets


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    params = FSTParams()
    params.NEGATIVE_SAMPLES = False
    ds = FstDataset(params)
    dl = DataLoader(
        dataset=ds,
        batch_size=64,
        collate_fn=ds.collate_fn
    )

    a_, b_, c_ = split_fst_dataset(ds, [0.1, 0.4, 0.5])

    for i, (sequence, label) in enumerate(dl):
        print(i, sequence, label)
    e = 0
