import abc
import logging
import random
from typing import Any, Dict, Generic, Iterable, List, Optional, Set, TypeVar

import numpy as np
from torch.utils.data import Dataset as TorchDataset

log = logging.getLogger(__name__)

DATA_DIR = ".data"

T = TypeVar("T")


class Dataset(TorchDataset, Iterable, Generic[T]):
    """Abstract Dataset.
    You should override this class to create a new dataset from a different source.
    All instances of the class return molecules (inputs, targets) as item.
    Note that some item might be None, if such an item comes out, the len(dataset)
    will change accordingly. This mean that len() is not constant and is only a
    prediction.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Optional[T]:
        """Return a tuple of molecules corresponding to (inputs, targets)."""
        ...

    def approx_defined(self) -> int:
        """Return the approx number of samples that are not None."""
        return len(self)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ListDataset(Dataset[T]):
    """Basic Dataset without any change to the inputs."""

    def __init__(self, samples: List[Optional[T]]):
        self.samples = samples

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Optional[T]:
        """Return the item at the index."""
        return self.samples[index]


class DictDataset(Dataset[T]):
    """Basic Dict-based dataset."""

    def __init__(self, samples: Dict[str, Any]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        """Return the number of samples in the Dataset."""
        # Todo a little brittle, refactor
        return len(self.samples["target"])

    def __getitem__(self, index: int) -> Optional[T]:
        """Return the item at the index."""
        sample = {
            "target": self.samples["target"][index],
            "mol": self.samples["mol"][index],
            "label": self.samples["label"][index],
        }

        sample["structure"] = self.samples["structures"][sample["target"]]

        return sample


class PreprocessedDataset(Dataset[T]):
    """Some item might be None."""

    def __init__(self, dataset: Dataset, filtered_func):
        self.dataset = dataset
        self.filtered_func = filtered_func

        self._removed_indexes: Set[int] = set()

    def approx_defined(self) -> int:
        """Return the approx number of samples that are not None."""
        return len(self.dataset) - len(self._removed_indexes)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Optional[T]:
        """Return the item at the index."""
        item = self.dataset[index]
        if item is None:
            return None

        item = self.filtered_func(item)
        if item is None:
            # We always store the positive index.
            if index < 0:
                index = len(self) - index
            self._removed_indexes.add(index)

        return item

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self[i]


class PartialDataset(Dataset[T]):
    """
    Wrapper over an existing dataset to only consider a fraction of it.
    Useful to use a fraction of a dataset for training and another for testing.
    """

    def __init__(self, dataset: Dataset[T], index_min: int, index_max: int):
        """Create the dataset.
        Note::
            `index_min` is included and `index_max` is excluded.
        """
        self.dataset = dataset
        self.index_min = index_min
        self.index_max = index_max

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.index_max - self.index_min

    def __getitem__(self, index: int) -> Optional[T]:
        if index < 0:
            raise ValueError(f"Negative index not supported yet {index}")

        index += self.index_min

        if index >= self.index_max:
            raise ValueError(
                f"Index {index} exceed the maximum index {self.index_max}"
            )

        if index < self.index_min:
            raise ValueError(
                f"Index {index} is below the minimum index {self.index_min}"
            )

        return self.dataset[index]


class ShuffledDataset(Dataset[T]):
    def __init__(self, dataset: Dataset[T]):
        self.dataset = dataset
        self.ids = np.arange(len(dataset))
        random.shuffle(self.ids)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Optional[T]:
        index = self.ids[index]
        return self.dataset[index]


class CacheDataset(Dataset[T]):
    """Cache Dataset to only do the preprocessing pipeline once per item.
    It will increase the RAM usage since the dataset will be twice in memory.
    """

    def __init__(self, dataset: Dataset[T]):
        self.dataset = dataset
        self.cache: Dict[int, Optional[T]] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Optional[T]:
        try:
            return self.cache[index]
        except KeyError:
            item = self.dataset[index]
            self.cache[index] = item
            return item


def split_dataset(dataset: Dataset, num_split: int):
    """Performs dataset splitting into num_split even splits"""
    dataset_size = len(dataset)
    datasets = []
    batch_size = int(dataset_size / num_split)
    current_index = 0
    while current_index < dataset_size:
        start_index = current_index
        end_index = current_index + batch_size
        end_index = min(end_index, dataset_size)
        datasets.append(PartialDataset(dataset, start_index, end_index))
        current_index += batch_size
    return datasets
