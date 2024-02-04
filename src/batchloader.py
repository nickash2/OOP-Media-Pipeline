import numpy as np
from typing import Generator


class BatchLoader:
    """
    A class used to create batches of data.

    ...

    Attributes
    ----------
    data : np.ndarray
        a numpy array containing the data to be batched
    batch_size : int
        the size of the batches to be created
    shuffle : bool, optional
        a flag used to determine whether to shuffle
        the data before batching (default is True)
    discard_last : bool, optional
        a flag used to determine whether to discard the
        last batch if it's smaller than batch_size (default is False)
    indices : np.ndarray
        a numpy array containing the indices of the data

    Methods
    -------
    __len__():
        Returns the number of batches that can be created
        from the data with the specified batch size.
    __iter__():
        Returns an iterator that yields batches of data.
    """
    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        discard_last: bool = False
                ) -> None:
        """
        Constructs all the necessary attributes for the BatchLoader object.

        Parameters
        ----------
            data : np.ndarray
                a numpy array containing the data to be batched
            batch_size : int
                the size of the batches to be created
            shuffle : bool, optional
                a flag used to determine whether to shuffle
                the data before batching (default is True)
            discard_last : bool, optional
                a flag used to determine whether to discard the
                last batch if it's smaller than batch_size (default is False)
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        if not isinstance(discard_last, bool):
            raise ValueError("discard_last must be a boolean")
        self._data = data
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._discard_last = discard_last
        self._indices = np.arange(data.shape[0])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._indices = np.arange(value.shape[0])  # Update indices

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value

    @property
    def discard_last(self):
        return self._discard_last

    @discard_last.setter
    def discard_last(self, value):
        self._discard_last = value

    @property
    def indices(self):
        return self._indices

    def __len__(self) -> int:
        """
        Returns the number of batches that can be
        created from the data with the specified batch size.

        Returns
        -------
        int
            the number of batches
        """
        if self._discard_last and self._data.shape[0] % self._batch_size != 0:
            return self._data.shape[0] // self.batch_size
        else:
            return ((self._data.shape[0] + self._batch_size - 1)
                    // self.batch_size)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        if self._shuffle:
            try:
                np.random.shuffle(self._indices)
            except Exception as e:
                raise RuntimeError(
                    "An error occurred while shuffling the indices"
                ) from e
        for i in range(0, len(self._indices), self.batch_size):
            batch_idx = self._indices[i: i + self.batch_size]
            if self._discard_last and len(batch_idx) < self.batch_size:
                break
            try:
                yield self._data[batch_idx]
            except Exception as e:
                raise RuntimeError(
                    "An error occurred while creating a batch"
                    ) from e
