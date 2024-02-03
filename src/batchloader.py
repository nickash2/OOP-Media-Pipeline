import numpy as np
from PIL import Image
import librosa
from typing import List, Tuple


class BatchLoader:
    """
    A class for loading data in batches.

    Args:
        data (np.ndarray): The list of data to load.
        batch_size (int): The size of each batch.
        file_type (str): The type of files to load ('image' or 'audio').
        shuffle (bool): Whether to shuffle the data.
        sequential (bool): Whether to load the data sequentially.

    """

    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        file_type: str,
        shuffle: bool = True,
        sequential: bool = False,
    ) -> None:
        self._check_type(batch_size, int, "batch_size")
        self._check_type(file_type, str, "file_type")
        self._check_type(shuffle, bool, "shuffle")
        self._check_type(sequential, bool, "sequential")
        self._check_type(data, np.ndarray, "data")
        self.batch_size = batch_size
        self.sequential = sequential
        self.shuffle = shuffle
        self.data = data
        self.num_batches = len(self.data) // self.batch_size
        self.current_index = 0
        self.indices = np.arange(len(self.data))

    def _randomize_batches(self) -> None:
        """
        Randomizes the order of the batches.
        """
        np.random.shuffle(self.indices)

    def _check_type(self, variable, variable_type, variable_name):
        """
        Checks if the variable has the correct type.

        Args:
            variable: The variable to check.
            variable_type: The expected type of the variable.
            variable_name: The name of the variable.

        Raises:
            TypeError: If the variable has an incorrect type.
        """
        if not isinstance(variable, variable_type):
            raise TypeError(f"{variable_name} must" +
                            "be a {variable_type.__name__}")

    def create_batches(self, discard_last_batch: bool = False) -> np.ndarray:
        """
        Creates the batches of data.

        Args:
            discard_last_batch (bool, optional): Whether to discard
            the last batch if its size is less than the batch size.

        Returns:
            numpy.ndarray: The batches of data.
        """
        if self.shuffle and not self.sequential:
            self._randomize_batches()

        if not discard_last_batch and len(self.data) % self.batch_size != 0:
            self.num_batches += 1

        if self.sequential:
            self.batches = [
                self.indices[i: i + self.batch_size]
                for i in range(0, len(self.data), self.batch_size)
            ]
        else:
            self.batches = np.array_split(self.indices, self.num_batches)

        return self.batches

    def __len__(self) -> int:
        """
        Returns the number of batches.

        Returns:
            int: The number of batches.
        """
        if len(self.data) % self.batch_size != 0:
            self.num_batches += 1
        return self.num_batches

    def __iter__(self) -> "BatchLoader":
        """
        Initializes the iterator.

        Returns:
            BatchLoader: The iterator object.
        """
        self.current_index = 0
        return self

    def __next__(self) -> List[np.ndarray]:
        """
        Loads the next batch of data.

        Returns:
            List[numpy.ndarray]: The batch of data.

        Raises:
            StopIteration: If there are no more batches to load.
        """
        if self.current_index >= len(self.batches):
            raise StopIteration

        batch_indices = self.batches[self.current_index]
        batch_data = [self._load_data(i) for i in batch_indices]

        self.current_index += 1
        return batch_data

    def _load_data(self,
                   index: int,
                   file_type: str
                   ) -> Image or Tuple[np.ndarray, int]:
        """
        Loads the data at the given index.

        Args:
            index (int): The index of the data.
            file_type (str): The type of file to load ('image' or 'audio').

        Returns:
            Image or Tuple[numpy.ndarray, int]: The loaded data.
        """
        if file_type == "image":
            return Image.open(self.data[index])
        elif file_type == "audio":
            return librosa.load(self.data[index])
