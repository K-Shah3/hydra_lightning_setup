import abc
from typing import Generic, List, TypeVar

import numpy as np

T = TypeVar("T")


class Feature(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def transform(self, batch: List[T]) -> np.ndarray:
        ...
