from torch import Tensor
from torch.utils.data import Dataset
from random import shuffle
import numpy as np
from fst_tools import FSTools
from rnn_models.dataset_params import FSTParams

PAD = "_PAD_"
END = "_END_"


class FstDoubleAcceptDataset(Dataset):
    def __init__(self, parmas: FSTParams, fst=None):
        self._fst = FSTools().rand_fst(parmas.FST_STATES_SIZE, parmas.FST_ALPHABET_SIZE, num_accept_states=2) \
            if fst is None else fst
        self._chr_embed = self._get_embeddings(self._fst.alphabet)  # index alphabet for embeddings
        self._data_one, self._data_two, self._acc_one, self._acc_two = self._build_data(parmas.DATASET_SIZE,  # get data
                                                                                        parmas.NEGATIVE_SAMPLES)
        self._mode = self._acc_one
        self._size_data = list(range(len(self._data_one)))

    def resize(self, num_samples):
        self._size_data = list(range(num_samples))

    def mode_one(self):
        self._mode = self._acc_one

    def mode_two(self):
        self._mode = self._acc_two

    def _get_embeddings(self, alphabet):
        embed = {symbol: i for i, symbol in enumerate(alphabet)}
        embed[END] = len(embed)                                     # special index for Start sequence
        embed[PAD] = len(embed)                                     # special index for padding
        return embed

    def _build_data(self, size, is_negative):
        negative_size = size // 2 if is_negative else 0
        positive_size = size - negative_size
        acc_one, acc_two = tuple(self._fst.accept_states())

        # add positive samples (sample , 1)
        positive_one, positive_two, negative_one, negative_two = [], [], [], []
        for positive_list, acc_type in [(positive_one, acc_one), (positive_two, acc_two)]:
            positive_set = set()
            temp_list = []
            while len(temp_list) < positive_size:
                # generate examples of type one
                sequence = self._fst.go()
                while self._fst.go(sequence)[0].id != acc_one:
                    sequence = self._fst.go()
                if str(sequence) not in positive_set:
                    positive_set.add(str(sequence))
                    temp_list.append([self._chr_embed[symbol] for symbol in sequence])

            for embed in list(set(tuple(x) for x in temp_list))[:positive_size]:
                positive_list.append((list(embed) + [self._chr_embed[END]], 1))

        # add negative samples (sample , 0)
        for positive_list, negative_list, acc_type in [(positive_one, negative_one, acc_one),
                                                       (positive_two, negative_two, acc_two)]:
            negative_set = set()
            temp_list = []
            i = 0
            while len(temp_list) < negative_size:
                sequence = self._fst.generate_relative_negative({acc_type}, sample_len=len(positive_list[i % len(positive_list)][0]) + 1)

                if str(sequence) not in negative_set:
                    negative_set.add(str(sequence))
                    temp_list.append([self._chr_embed[symbol] for symbol in sequence])
                    i += 1

            # add negative samples (sample , 0)
            for embed in list(set(tuple(x) for x in temp_list))[:negative_size]:
                negative_list.append((list(embed) + [self._chr_embed[END]], 0))

        data_one = negative_one + positive_one
        data_two = negative_two + positive_two
        shuffle(data_one)
        shuffle(data_two)
        return data_one, data_two, acc_one, acc_two

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
        flag_out_of_range = self._size_data[index]
        return self._data_one[index] if self._mode == self._acc_one else self._data_two[index]

    def __len__(self):
        return len(self._size_data)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    params = FSTParams()
    params.NEGATIVE_SAMPLES = True
    ds = FstDoubleAcceptDataset(params)
    dl = DataLoader(
        dataset=ds,
        batch_size=100,
        collate_fn=ds.collate_fn
    )

    for i_, (sequence_, label_) in enumerate(dl):
        print(i_, sequence_, label_)
    e = 0
