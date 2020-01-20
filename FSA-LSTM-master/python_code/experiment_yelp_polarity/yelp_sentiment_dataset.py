import string
from random import shuffle

from torch import Tensor
from torch.utils.data import Dataset
SOS, EOS, UNK = 0, 1, 2


class YelpSentimentDataset(Dataset):
    def __init__(self, sources: list, ready=None):
        if ready is not None:
            self._data = ready['data']
            return
        self._sources = sources
        self._data, self._word_to_idx, self._idx_to_word = self._build_data()

    @property
    def num_words(self):
        return len(self._idx_to_word)

    def _build_data(self):
        sent_data, data, idx_to_word = [], [], []
        words_to_idx = {'SOS': SOS, 'EOS': EOS, 'UNK': UNK}
        # read data as sentence at lower case & without punctuation
        for source in self._sources:
            sent_data += self._read_source(source)
        # process sentences from text to numerical values || prepare word_to_idx string
        for txt_sent, sentiment in sorted(sent_data, key=lambda x: x[0]):
            numeric_sent = []
            for word in txt_sent:
                # get word index or add it to dictionary
                idx = words_to_idx.get(word, len(words_to_idx))
                words_to_idx[word] = idx
                # add word as number to the sentence
                numeric_sent.append(idx)
            # add sentence to the data
            data.append((numeric_sent, sentiment))
        # create idx_to_word list
        idx_to_word = [word for word, i in sorted(words_to_idx.items(), key=lambda x: x[1])]
        return data, words_to_idx, idx_to_word

    def _read_source(self, source):
        data = []
        source_file = open(source)
        # read file
        for sample_row in source_file:
            # remove punctuation & lower
            sentiment, sent = sample_row.split(",", 1)
            sentiment = int(sentiment[1]) - 1
            sent = str(sent).translate(str.maketrans('', '', string.punctuation)).lower().split()
            data.append((sent, sentiment))
        return data

    def collate_fn(self, batch):
        max_len = len(max(batch, key=lambda x: len(x[0]))[0])
        batch_sent, label_batch = [], []
        for sent, label in batch:
            label_batch.append(label)
            batch_sent.append([SOS] + sent + [EOS] * (max_len + 1 - len(sent)))
        return Tensor(batch_sent).long(), Tensor(label_batch).long()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


def yelp_sentiment_data_split(dataset, split_list):
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
            "data": new_data[i]
        }
        sub_datasets.append(YelpSentimentDataset([], ready=ready_dict))
    return sub_datasets


if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    sources_ = [
        os.path.join("yelp_review_polarity_csv", "train.csv"),
        os.path.join("yelp_review_polarity_csv", "test.csv"),
    ]
    ds = YelpSentimentDataset(sources_)
    dl = DataLoader(ds,
                    batch_size=64,
                    collate_fn=ds.collate_fn,
                    shuffle=True)
    for i, (sample_, label_) in enumerate(dl):
        print(sample_, label_)
