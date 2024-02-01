import numpy as np
from PIL import Image
import librosa
from typing import List, Tuple


class BatchLoader:
    def __init__(self,
                 batch_size: int,
                 file_type: str,
                 shuffle: bool = True,
                 data: list = None,
                 sequential: bool = False) -> None:
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if not isinstance(file_type, str):
            raise TypeError("file_type must be a string")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean")
        if not isinstance(sequential, bool):
            raise TypeError("sequential must be a boolean")
        if data is not None and not isinstance(data, list):
            raise TypeError("data must be a list")
        self.batch_size = batch_size
        self.sequential = sequential
        self.shuffle = shuffle
        self.data = data
        self.num_batches = len(self.data) // self.batch_size
        self.current_index = 0
        self.indices = np.arange(len(self.data))

    def _randomize_batches(self) -> None:
        np.random.shuffle(self.indices)

    def create_batches(self, discard_last_batch: bool = False) -> np.ndarray:
        if self.shuffle and not self.sequential:
            self._randomize_batches()

        if not discard_last_batch and len(self.data) % self.batch_size != 0:
            self.num_batches += 1

        if self.sequential:
            self.batches = [self.indices[i:i+self.batch_size]
                            for i in range(0, len(self.data), self.batch_size)]
        else:
            self.batches = np.array_split(self.indices, self.num_batches)

        return self.batches

    def __len__(self) -> int:
        if len(self.data) % self.batch_size != 0:
            self.num_batches += 1
        return self.num_batches

    def __iter__(self) -> 'BatchLoader':
        self.current_index = 0
        return self

    def __next__(self) -> List[np.ndarray]:
        if self.current_index >= len(self.batches):
            raise StopIteration

        batch_indices = self.batches[self.current_index]
        batch_data = [self.load_data(i) for i in batch_indices]

        self.current_index += 1
        return batch_data

    def _load_data(self, index: int,
                   file_type: str
                   ) -> Image or Tuple[np.ndarray, int]:
        if (file_type == 'image'):
            return Image.open(self.data[index])
        elif (file_type == 'audio'):
            return librosa.load(self.data[index])
