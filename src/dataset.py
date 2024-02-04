import os
from typing import Dict, Tuple, Any, Generator, List
from abstract_dataset import AbstractDataset
import librosa
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools


class Dataset(AbstractDataset):
    def __init__(self, root: str, data_type: str) -> None:
        """
        Initialize a Dataset.

        Args:
        root (str): The path to the directory containing the data files.
        data_type (str): The type of data in the dataset.
        """
        if not isinstance(root, str):
            raise TypeError("root must be a string")
        if not isinstance(data_type, str):
            raise TypeError("data_type must be a string")

        super().__init__(root, data_type)
        try:
            self.data_paths = [os.path.join(root, file)
                               for file in os.listdir(root)]
        except FileNotFoundError:
            raise FileNotFoundError(f"No such directory: {root}")

    def _load_labels(self) -> Dict[str, Any]:
        """
        Load the labels for the dataset.

        Returns:
        dict: A dictionary mapping filenames to labels.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
        int: The number of data files in the dataset.
        """
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Tuple[np.array, str]:
        """
        Get the data and label at the given index.

        Args:
        idx (int): The index of the data point.

        Returns:
        tuple: A tuple containing the data and label.
        """
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")

        # Use the lazy loading function to get the data at the given index
        data, label = next(
            itertools.islice(self.load_data_lazy(), idx, idx + 1)
            )

        return data, label

    def __iter__(self):
        return self.load_data_lazy()

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

    def _load_data(self, file_path: str) -> np.array | Tuple[np.array, int]:
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

    def split(self, ratio: float) -> Tuple[np.array, np.array]:
        """
        Split the dataset into training and testing sets.

        Args:
            ratio (float): The ratio of testing data to total data.

        Returns:
            tuple: A tuple containing the training data and testing data.
        """
        if self.data is None:
            self.load_data_eager()
        files = list(self.data.keys())
        train_files, test_files = train_test_split(files, test_size=ratio)
        train_data = [self.data[file] for file in train_files]
        test_data = [self.data[file] for file in test_files]
        return np.array(train_data), np.array(test_data)


class LabeledDataset(Dataset):
    """
    A dataset where each data point has a label.
    The labels are loaded from a separate CSV file.
    """

    def __init__(self, root: str, label_file: str, data_type: str) -> None:
        """
        Initialize a LabeledDataset.

        Args:
        root (str): The path to the directory containing the data files.
        label_file (str): The path to the CSV file containing the labels.
        data_type (str): The type of data in the dataset.
        """
        self.label_file = label_file
        super().__init__(root, data_type)

    def _load_labels(self) -> Dict[str, str]:
        """
        Load labels from the CSV file.
        Each line in the file should be formatted as '<filename>,<label>'.

        Returns:
        dict: A dictionary mapping filenames to labels.
        """
        path = os.path.join(self.root, self.label_file)
        try:
            with open(path, "r"):
                df = pd.read_csv(path, header=None)
                # Remove the .jpg extension from the first column
                df[0] = df[0].str.replace(".jpg", "")
                # Sort the DataFrame by the first column
                df = df.sort_values(by=[0])
                # Convert the DataFrame to a dictionary and return it
                return df.set_index(0)[1].to_dict()
        except FileNotFoundError:
            raise FileNotFoundError(f"No such file: {path}")

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
        int: The number of data files in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.array, List[str]]:
        """
        Get the data and label at the given index.

        Args:
        idx (int): The index of the data point.

        Returns:
        tuple: A tuple containing the data and label.
        """
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")
        values, _ = super().__getitem__(idx)
        return values, list(self.labels.values())[idx]

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
            self.labels[file_name_without_extension],
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
                    print(f"Loading data for file {file_name}")
                    data = self._load_data(file_path)
                    file_name_no_extension, _ = os.path.splitext(file_name)
                    yield (np.array(data), self.labels[file_name_no_extension])


class UnlabeledDataset(Dataset):
    """
    A dataset where each data point does not have a label.
    """

    def __init__(self, root: str, data_type: str) -> None:
        """
        Initialize an UnlabeledDataset.

        Args:
        root (str): The path to the directory containing the data files.
        data_type (str): The type of data in the dataset.
        """
        super().__init__(root, data_type)

    def _load_labels(self) -> Dict[str, None]:
        """
        Since the dataset is unlabeled,
        assign None as the label for each data file.

        Returns:
        dict: A dictionary mapping filenames to None.
        """
        return {file: None for file in os.listdir(self.root)}


class HierarchicalDataset(Dataset):
    """
    A dataset where each data point has a label.
    The labels are determined based on the directory structure.
    Each subdirectory in the root directory represents a class,
    and the files in each subdirectory are the data points for that class.
    """

    def __init__(self, root: str, data_type: str) -> None:
        """
        Initialize a HierarchicalLabeledDataset.

        Args:
        root (str): The path to the directory containing the data files.
        data_type (str): The type of data in the dataset.
        """
        super().__init__(root, data_type)

    def _load_labels(self) -> Dict[str, List[str]]:
        """
        Load the labels from the dataset root directory.

        Returns:
            A dictionary containing class names as keys and a
            sorted list of labels as values.
        Raises:
            FileNotFoundError: If the dataset root directory does not exist.
        """
        labels = {}
        try:
            for class_folder in os.listdir(self.root):
                class_path = os.path.join(self.root, class_folder)
                if os.path.isdir(class_path):
                    labels[class_folder] = sorted(os.listdir(class_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"No such directory: {self.root}")
        return labels

    def __len__(self) -> int:
        """
        Returns the total number of data points in the dataset.

        If the data is not loaded, it will be loaded
        eagerly before calculating the length.

        Returns:
            int: The total number of data points in the dataset.
        """
        if self.data is None:
            self.load_data_eager()
        return sum(len(data_points) for data_points in self.data.values())

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Get the item at the given index.

        Args:
            idx (int or tuple): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the data point and the label.

        Raises:
            TypeError: If idx is not an integer or a tuple of integers.
            TypeError: If idx is a tuple and not all elements are integers.
            IndexError: If idx is an integer and is out of range.

        """
        if self.data is None:
            self.load_data_eager()
        if not isinstance(idx, (int, tuple)):
            raise TypeError("idx must be an integer or a tuple of integers")
        if isinstance(idx, tuple) and not all(isinstance(i, int) for i in idx):
            raise TypeError("all elements of idx must be integers")
        if isinstance(idx, int) and not (0 <= idx < len(self.data)):
            raise IndexError("idx out of range")

        self.keys = list(self.data.keys())
        # If idx is a tuple, treat the first element as the directory index
        # and the second element as the data point index within that directory.
        if isinstance(idx, tuple):
            directory_index, data_point_index = idx
            directory_key = self.keys[directory_index]
            # returns a tuple of the data point and the label
            return self.data[directory_key][data_point_index], directory_key

        else:
            # If idx is not a tuple, it is a direct index to the data points.
            return list(self.data.values())[idx], self.keys[idx]

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
        return np.array(self.data), self.labels

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
