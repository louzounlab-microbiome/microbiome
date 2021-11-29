from torch.utils.data import Dataset
import torch
import pandas as pd
from typing import Callable, Any, List
import pytorch_lightning as pl
from sklearn.model_selection import GroupShuffleSplit
"""A simple dataset class which supports supervised learning i.e the dataset consists of a dataframe X and labels y
The dataset additionally supports transforms which will be applied on both X and y. """


class CustomDataset(Dataset):
    def __init__(self, x_df: pd.DataFrame, y_target: pd.Series, transform: Callable[[Any, Any], Any] = None):
        self.x = x_df
        self.y = y_target
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


class LearningDataSet(pl.LightningDataModule):
    def __init__(self, data: pd.DataFrame, target: pd.Series,group_id=None, train_test_ratio=0.8, train_validation_ratio=0.8,
                 train_batch_size=10, validation_batch_size=10, transform: Callable[[Any, Any], Any] = None):
        super().__init__()
        self.data = CustomDataset(data, target, transform)
        self.group_id=group_id
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.train_test_ratio = train_test_ratio
        self.train_validation_ratio = train_validation_ratio
        self.transform = transform

    def setup(self, stage: str = None):
        if stage == 'fit':
        # train and validate only according to the patients with both fields.
            if self.group_id is None:
                train_and_validation_size = int(self.train_test_ratio * len(self.data))
                test_size = len(self.data) - train_and_validation_size
                train_validation_data, self.test_data = torch.utils.data.random_split(self.data,
                                                                                  [train_and_validation_size, test_size])
            else:
                gss = GroupShuffleSplit(n_splits=1, train_size=self.train_test_ratio)
                train_validation_idx, test_idx = next(gss.split(self.data, groups=self.group_id))
                train_validation_data = torch.utils.data.Subset(self.data,train_validation_idx)
                self.test_data = torch.utils.data.Subset(self.data,test_idx)

            train_size = int(self.train_validation_ratio * len(train_validation_data))
            validation_size = len(train_validation_data) - train_size
            self.train_data, self.validation_data = torch.utils.data.random_split(train_validation_data,
                                                                                  [train_size, validation_size])
        else:
            pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LearningDataSet")
        parser.add_argument('--train_test_ratio', type=float,default = 0.8)
        parser.add_argument('--train_validation_ratio', type=float,default = 0.8)
        parser.add_argument('--train_batch_size', type=int, default= 16)
        parser.add_argument('--validation_batch_size',type=int, default= 10)


        return parent_parser

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, shuffle=True,
                                           batch_size=self.train_batch_size, )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validation_data, batch_size=self.validation_batch_size,
                                           shuffle=False)

    def test_dataloader(self):
        # Notice that the test data is loaded as one unit.
        return torch.utils.data.DataLoader(self.test_data, shuffle=False, batch_size=len(self.test_data))
