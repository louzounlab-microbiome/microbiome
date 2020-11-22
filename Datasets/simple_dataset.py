from torch.utils.data import Dataset
import torch
import pandas as pd
from typing import Callable, Any, List

"""A simple dataset class which supports supervised learning i.e the dataset consists of a dataframe X and labels y
The dataset additionally supports transforms which will be applied on both X and y. """


class CustomDataset(Dataset):
    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame, transform: Callable[[Any, Any], Any] = None):
        self.x = x_df
        self.y = y_df
        self.transform = transform

    def __getitem__(self, index: int):
        """
        When indexing the dataset,a tuple describing the corresponding row and its label will be returned,
        if transform is not None the function will be applied on the corresponding row and its label.
        """

        if self.transform is None:
            return self.x.iloc[index], self.y.iloc[index]
        return self.transform(self.x.iloc[index].values, self.y.iloc[index])

    def __len__(self) -> int:
        return len(self.x)


class ToTensor(object):
    """Convert all inputs into tensor form."""
    """A transformer class, which its constructor receives as an input a list of tensor types """

    def __init__(self, type_list: List[Any] = None):
        self.types = type_list

    def __call__(self, *args):
        """
        If type list is given, the transformer transforms each argument into a tensor with a type according to type list.
        otherwise, no casting is done.
        """
        if self.types:
            return [torch.tensor(arg).type(ty) for arg, ty in zip(args, self.types)]
        return [torch.tensor(arg) for arg in args]
