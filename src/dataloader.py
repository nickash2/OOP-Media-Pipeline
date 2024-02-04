import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Generator, List
import numpy as np
from PIL import Image
import librosa


class DataLoader(ABC):
    def __init__(
        self,
        root: str,
        data_type: str,
        labels=None
            ) -> None:
        """
        Initialize the DataLoader class.

        Args:
            root (str): The root directory path.
            data_type (str): The type of data.
            labels (list, optional): The labels for the data. Defaults to None.
        """
        self.root = root
        self.data_type = data_type
        self.data_paths = [
            os.path.join(self.root, dir_path)
            for dir_path in sorted(os.listdir(self.root))
        ]
        self.labels = labels

    @abstractmethod
    def load_data_eager(self, path: str) -> None:
        pass

    @abstractmethod
    def load_data_lazy(self, path: str) -> None:
        pass

    def _load_data(self, file_path: str) -> None:
        """
        Load a data point from a file.

        Args:
        file_path (str): The path to the data file.

        Returns:
        Any: The loaded data point.
        """
        if self.data_type == "image":
            img = Image.open(file_path)
            return np.array(img.convert("RGB"))  # use rgb
        elif self.data_type == "audio":
            audio, sr = librosa.load(file_path)
            return (audio, sr)


class UnlabeledDataLoader(DataLoader):
    def __init__(self, root: str, data_type) -> None:
        """
        Initialize the UnlabeledDataLoader.

        Args:
            root (str): The root directory of the dataset.
            data_type: The type of data to load.

        Returns:
            None
        """
        self.data_type = data_type
        super().__init__(root, data_type)

    def load_data_eager(self) -> Dict[str, np.array]:
        """
        Load all data points in the dataset eagerly.

        Returns:
            dict: A dictionary mapping filenames to data points.
        """
        self.data = {}
        for dir_path in self.data_paths:
            if os.path.isdir(dir_path):
                for file_name in sorted(os.listdir(dir_path)):
                    file_path = os.path.join(dir_path, file_name)
                    self.data[file_name] = self._load_data(file_path)
        if self.data_type == "image":
            return np.array(list(self.data.values()))
        else:
            data_list = list(self.data.values())
            audio_data, sampling_rates = zip(*data_list)
            return np.array(audio_data, dtype=object), np.array(sampling_rates)

    def load_data_lazy(self) -> Generator:
        """
        Load data points in the dataset lazily.

        Yields:
            Any: The next data point in the dataset.
        """
        for dir_path in self.data_paths:
            if os.path.isdir(dir_path):
                for file_name in sorted(os.listdir(dir_path)):
                    file_path = os.path.join(dir_path, file_name)
                    yield self._load_data(file_path)


class LabeledDataLoader(DataLoader):
    def __init__(
        self,
        root: str,
        data_type: str,
        labels: List[str]
    ) -> None:
        """
        Initialize the LabeledDataLoader.

        Args:
            root (str): The root directory of the data.
            data_type (str): The type of data.
            labels (List[str]): The list of labels.

        Returns:
            None
        """
        self.data_type = data_type
        super().__init__(root, data_type, labels)

    def load_data_eager(self) -> Tuple[Dict[str, np.array], Dict[str, str]]:
        """
        Load the data and labels eagerly.

        Returns:
            tuple: A tuple containing the data and labels.
        """
        self.data = {}
        for dir_path in self.data_paths:
            if os.path.isdir(dir_path):
                for file_name in sorted(os.listdir(dir_path)):
                    file_path = os.path.join(dir_path, file_name)
                    self.data[file_name] = self._load_data(file_path)
        file_name_without_extension, _ = os.path.splitext(file_name)
        return (
            np.array(list(self.data.values()), dtype=object),
            self.labels,
        )

    def load_data_lazy(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        """
        Load the data and labels lazily.

        Yields:
            tuple: A tuple containing the data and label.
        """
        for dir_path in self.data_paths:
            if os.path.isdir(dir_path):
                for file_name in sorted(os.listdir(dir_path)):
                    file_path = os.path.join(dir_path, file_name)
                    data = self._load_data(file_path)
                    file_name_no_extension, _ = os.path.splitext(file_name)
                    yield (np.array(data), self.labels[file_name_no_extension])


class HierarchicalDataLoader(DataLoader):
    def __init__(self, root: str, data_type: str, labels: List[str]):
        """
        Initialize the HierarchicalDataLoader.

        Args:
            root (str): The root directory path.
            data_type (str): The type of data.
            labels (List[str]): The list of labels.
        """
        self.data_type = data_type
        super().__init__(root, data_type, labels)

    def load_data_eager(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load the data eagerly into memory.

        Returns:
            Tuple[np.ndarray, List[str]]: A tuple containing the loaded
            data and the corresponding labels.
        """
        self.data = {}
        for class_folder in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                # If current entry is a directory, use its name as the label
                self.data[class_folder] = []
                for data_file in sorted(os.listdir(class_path)):
                    file_path = os.path.join(class_path, data_file)
                    self.data[class_folder].append(self._load_data(file_path))
        return (np.array(list(self.data.values()), dtype=object), self.labels)

    def load_data_lazy(self) -> Generator[Tuple[np.ndarray, str], None, None]:
        """
        Load the data lazily as a generator.

        Yields:
            Generator[Tuple[np.ndarray, str], None, None]: A generator
            that yields tuples of loaded data and corresponding labels.
        """
        for class_folder in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                # If current entry is a directory, use its name as the label
                for data_file in sorted(os.listdir(class_path)):
                    file_path = os.path.join(class_path, data_file)
                    data = self._load_data(file_path)
                    yield (np.array(data), self.labels[data_file])
