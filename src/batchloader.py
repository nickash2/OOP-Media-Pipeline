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
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.discard_last = discard_last
        self.indices = np.arange(data.shape[0])

    def __len__(self) -> int:
        """
        Returns the number of batches that can be
        created from the data with the specified batch size.

        Returns
        -------
        int
            the number of batches
        """
        if self.discard_last and self.data.shape[0] % self.batch_size != 0:
            return self.data.shape[0] // self.batch_size
        else:
            return ((self.data.shape[0] + self.batch_size - 1)
                    // self.batch_size)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        if self.shuffle:
            try:
                np.random.shuffle(self.indices)
            except Exception as e:
                raise RuntimeError(
                    "An error occurred while shuffling the indices"
                ) from e
        for i in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[i: i + self.batch_size]
            if self.discard_last and len(batch_idx) < self.batch_size:
                break
            try:
                yield self.data[batch_idx]
            except Exception as e:
                raise RuntimeError(
                    "An error occurred while creating a batch"
                    ) from e
