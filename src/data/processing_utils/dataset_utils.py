import math
import random
from enum import Enum
from copy import deepcopy
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)
import numpy as np
from omegaconf import DictConfig
from src.data.processing_utils.base import Dataset, CacheDataset
from typing import List, Optional, Tuple
from src.data.processing_utils import pdbbind_utils
from torch.utils.data import DataLoader
from src.data.features.base import Feature
import rdkit

O = TypeVar("O")

class BatchType(Enum):
    ITEM = "item"
    PAIR = "pair"

def prepare_dataset(
    config: DictConfig
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """
    Prepare all the dataset from the experiment config.
    The preprocessing is applied here, no need to do it after.
    """
    train_dataset, validation_dataset, test_dataset = load(
        config=config.dataset
    )
    # Todo preprocessing
    return train_dataset, validation_dataset, test_dataset

def load(
    config: DictConfig,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the dataset from the config. Each dataset requires a specific parsing script stored in /source
    :param config: The dataset config
    :type config: DatasetConfig
    :returns: The train, valid and test datasets.
    """
    # log.info(f"Loading dataset: {config.dataset.source}")
    print(config)
    if config.source == "pdbbind":
        return pdbbind_utils.load(config)
    else:
        raise ValueError(f"Dataset source ({config.source}) is not supported.")

def prepare_dataloader(
    config: DictConfig,
    feature_inputs,
    train_dataset: Optional[Dataset],
    valid_dataset: Optional[Dataset],
    test_dataset: Optional[Dataset],
    output_dir=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Todo move to config
    generate_batch_size_ratio = 0.20
    valid_batch_size_ratio = generate_batch_size_ratio

    if train_dataset is not None:
        # log.info("Preparing train dataloader ...")
        train_dataloader = create_dataloader(
            config.dataloader,
            feature_inputs,
            train_dataset,
            data_usage_ratio=config.dataloader.data_usage_ratio,
        )
    else:
        # log.info(
        #     "Train dataset not specified. Skipping dataloader preparation"
        # )
        train_dataloader = None

    if valid_dataset is not None:
        # log.info("Preparing valid dataloader ...")
        valid_dataloader = create_dataloader(
            config.dataloader,
            feature_inputs,
            valid_dataset,
            batch_size_ratio=valid_batch_size_ratio,
            data_usage_ratio=config.dataloader.data_usage_ratio,
        )
    else:
        # log.info(
        #     "Validation dataset not specified. Skipping dataloader preparation"
        # )
        valid_dataloader = None

    if test_dataset is not None:
        # log.info("Preparing test dataloader ...")
        test_dataloader = create_dataloader(
            config.dataloader,
            feature_inputs,
            test_dataset,
            batch_size_ratio=generate_batch_size_ratio,
            data_usage_ratio=config.dataloader.data_usage_ratio,
        )
    else:
        # log.info("Test dataset not specified. Skipping dataloader preparation")
        test_dataloader = None

    return train_dataloader, valid_dataloader, test_dataloader

def create_dataloader(
    config: DictConfig,
    feature_inputs: Optional[List[Feature]],
    dataset: Dataset,
    batch_size_ratio: float = 1.0,
    data_usage_ratio: float = 1.0,
) -> DataLoader:
    dataloader = create_single_process_dataloader(
        config,
        feature_inputs,
        dataset,
        batch_size_ratio=batch_size_ratio,
        data_usage_ratio=data_usage_ratio,
    )

    return dataloader

def create_single_process_dataloader(
    config: DictConfig,
    feature_inputs: Dict[str, List[Feature]],
    dataset: Dataset,
    batch_size_ratio: float = 1.0,
    data_usage_ratio: float = 1.0,
) -> DataLoader:
    """
    Creates a single process dataloader for a dataset from a given dataloader config and feature transforms
    """
    return DataLoader(
        dataset,
        math.floor(config.batch_size * batch_size_ratio),
        config.shuffle,
        feature_inputs,
        data_usage_ratio=data_usage_ratio,
        batch_type=config.batch_type,
    )

class MultiProcessException(Exception):
    """Error message for multiprocessing dataloader"""

    def __init__(self, e: Exception):
        super(MultiProcessException, self).__init__(
            f"An error occurred on dataloader process: Error: {e}"
        )


class DataLoader(Iterable, Generic[O]):
    """Load the data from a dataset.
    The dataloader convert a dataset of molecules into batches of np.ndarray
    corresponding to the inputs and target features.
    Notes:
        If the dataset can't be perfectly divided by the batch size, the last
        batch will be smaller. It ensures that all samples are used.
    """

    def __init__(
        self,
        dataset: Dataset[Tuple[str, rdkit.Chem.Mol, O]],
        batch_size: int,
        shuffle: bool,
        inputs_features: Dict[str, Optional[List]],
        data_usage_ratio: float = 1.0,
        batch_type: BatchType = BatchType.ITEM,
        cache_enable: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inputs_features = inputs_features
        self.data_usage_ratio = data_usage_ratio
        self.cache_enable = cache_enable
        self.batch_type = batch_type

        # Select batch type
        if BatchType(batch_type) == BatchType.ITEM:
            self._batches = self._batches_items
        else:
            raise ValueError(f"Batch type {batch_type} not supported")

        self._ids = np.arange(len(self.dataset))

        if self.cache_enable:
            self.dataset = CacheDataset(self.dataset)

        if self.shuffle:
            random.shuffle(self._ids)

        if not self.inputs_features:
            message = "No input features, cannot load data."
            # log.error(message)
            raise ValueError(message)

    def __len__(self) -> int:
        """Return the number of batches in the dataloader."""
        return int(
            len(self.dataset) * self.data_usage_ratio / (self.batch_size)
        )

    def approx_defined(self) -> int:
        return int(self.dataset.approx_defined() * self.data_usage_ratio)

    def __iter__(self):
        """Iterate over the dataset with batches.
        Return: X features and Y.
                If there are multiple features, a list of Tensor is returned.
        """
        batches = self._batches(self._ids)
        return self._create_iterable(batches)

    def sample(self, num_batches: int):
        """Sample randomly a num_batches of items into the dataset."""
        ids = deepcopy(self._ids)
        random.shuffle(ids)
        batches = self._batches(ids)
        return self._create_iterable(batches, max_batches=num_batches)

    def _create_iterable(self, batches, max_batches: int = None):
        """Creates an iterable object from batch by applying featurisation transforms"""
        count = 0

        for batch_x_a, batch_x_b, batch_y, batch_x_c in batches:
            # batch_x_a was just the name
            # batch_x_b mol of ligand
            # batch_y y affinity label 
            # batch_x_c structure - target + ligand coords and the target types 
            if max_batches is not None and count >= max_batches:
                break

            x_a = [
                feature.transform(batch_x_c, batch_x_a)
                for feature in self.inputs_features["protein"]
            ]

            x_b = [
                feature.transform(batch_x_b)
                for feature in self.inputs_features["ligand"]
            ]

            Y = batch_y

            yield x_a, x_b, Y

            count += 1

    def _num_tokens(self, mol):
        tokens = self.tokenizer.tokenize(mol)
        return len(tokens)

    def _batches_dynamic(self, ids: List[int]):
        id_index = -1
        while True:
            batch_x_a: List[Any] = []
            batch_x_b: List[Any] = []
            batch_y: List[Any] = []

            max_length = 0
            batch_size = 0
            while batch_size < self.batch_size:
                id_index += 1

                if id_index >= len(ids) * self.data_usage_ratio:
                    if batch_x_a:
                        yield batch_x_a, batch_x_b, batch_y
                    if batch_x_b:
                        yield batch_x_a, batch_x_b, batch_y
                    return

                sample_id = ids[id_index]
                item = self.dataset[sample_id]

                if item is None:
                    continue
                # print(f"Dataloader item: {item}")
                x_a = item["target"]
                x_b = item["mol"]
                y = item["label"]

                seq_length_a = self._seq_length(x_a)
                seq_length_b = self._seq_length(x_b)

                if seq_length_a > self.batch_size:
                    log.warning(
                        f"Batch size smaller than one single item, skipping {self.batch_size} < {seq_length_a}. Interactor A"
                    )
                    continue
                if seq_length_b > self.batch_size:
                    log.warning(
                        f"Batch size smaller than one single item, skipping {self.batch_size} < {seq_length_b}. Interactor B"
                    )
                    continue

                max_length = max(max_length, seq_length_a, seq_length_b)
                # Todo check this holds with batch_x_b
                batch_size = (1 + len(batch_x_a)) * max_length

                if batch_size > self.batch_size:
                    id_index -= 1
                else:
                    batch_x_a.append(x_a)
                    batch_x_b.append(x_b)
                    batch_y.append(y)

            yield batch_x_a, batch_x_b, batch_y

    def _batches_items(
        self, ids: List[int]
    ):
        id_index = -1
        while True:
            batch_x_a: List[str] = []
            batch_x_b: List[rdkit.Chem.Mol] = []
            batch_x_c: List[O] = []
            batch_y: List[O] = []

            while len(batch_y) != self.batch_size:
                id_index += 1

                if id_index >= len(ids) * self.data_usage_ratio:
                    if batch_x_a:
                        yield batch_x_a, batch_x_b, batch_y, batch_x_c
                    if batch_x_b:
                        yield batch_x_a, batch_x_b, batch_y, batch_x_c
                    return

                sample_id = ids[id_index]
                item = self.dataset[sample_id]

                if item is None:
                    continue

                batch_x_a.append(item["target"])
                batch_x_b.append(item["mol"])
                batch_y.append(item["label"])
                batch_x_c.append(item["structure"])

            yield batch_x_a, batch_x_b, batch_y, batch_x_c

