import csv
import os
from typing import Dict, Tuple, Any, Generator
from abstract_dataset import AbstractDataset
import librosa
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class Dataset(AbstractDataset):
    def __init__(self, root: str, data_type: str) -> None:
        """
        Initialize a Dataset.

        Parameters:
        root (str): The path to the directory containing the data files.
        data_type (str): The type of data in the dataset.
        """
        super().__init__(root, data_type)
        self.data_paths = [os.path.join(root, file)
                           for file in os.listdir(root)]

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
        int: The number of data points in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Any:
        """
        Get a specific data point from the dataset.

        Parameters:
        idx (int): The index of the data point to retrieve.

        Returns:
        Any: The data point at the specified index.
        """
        if self.data is None:
            self.load_data_eager()
        return list(self.data.values())[idx]

    def load_data_eager(self) -> Dict[str, Any]:
        """
        Load all data points in the dataset eagerly.

        Returns:
        dict: A dictionary mapping filenames to data points.
        """
        self.data = {}
        for dir_path in self.data_paths:
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    self.data[file_name] = self._load_data(file_path)
        return self.data

    def load_data_lazy(self) -> Generator[Any, None, None]:
        """
        Load data points in the dataset lazily.

        Yields:
        Any: The next data point in the dataset.
        """
        for dir_path in self.data_paths:
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    yield self._load_data(file_path)

    def _load_data(self, file_path: str) -> Any:
        """
        Load a data point from a file.

        Parameters:
        file_path (str): The path to the data file.

        Returns:
        Any: The loaded data point.
        """
        if self.data_type == 'image':
            img = Image.open(file_path)
            return np.array(img.convert("RGB"))
        elif self.data_type == 'audio':
            audio, sr = librosa.load(file_path)
            return (audio, sr)

    def split(self, ratio: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split the dataset into training and testing sets.

        Parameters:
        ratio (float): The ratio of testing data to total data.

        Returns:
        tuple: A tuple containing the training data and testing data.
        """
        if self.data is None:
            self.load_data_eager()
        files = list(self.data.keys())
        train_files, test_files = train_test_split(files, test_size=ratio)
        train_data = {file: self.data[file] for file in train_files}
        test_data = {file: self.data[file] for file in test_files}
        return train_data, test_data


class LabeledDataset(Dataset):
    """
    A dataset where each data point has a label.
    The labels are loaded from a separate CSV file.
    """

    def __init__(self, root: str, label_file: str, data_type: str) -> None:
        """
        Initialize a LabeledDataset.

        Parameters:
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
        with open(path, "r") as f:
            reader = csv.reader(f)
            # Dictionary where the keys are filenames and the values are labels
            labels = {rows[0]: rows[1] for rows in reader}
        return labels

    def load_data_eager(self):
        self.data = {}
        for dir_path in self.data_paths:
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    self.data[file_name] = self._load_data(file_path)
        return self.data, self.labels[file_name]

    def load_data_lazy(self):
        for dir_path in self.data_paths:
            if os.path.isdir(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    self.data[file_name] = self._load_data(file_path)
                    yield self.data[file_name], self.labels[file_name]


class UnlabeledDataset(Dataset):
    """
    A dataset where each data point does not have a label.
    """

    def __init__(self, root: str, data_type: str) -> None:
        """
        Initialize an UnlabeledDataset.

        Parameters:
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

        Parameters:
        root (str): The path to the directory containing the data files.
        data_type (str): The type of data in the dataset.
        """
        super().__init__(root, data_type)

    def _load_labels(self) -> Dict[str, str]:
        """
        Load labels based on the directory structure.

        Returns:
        dict: A dictionary mapping filenames to labels.
        """
        labels = {}
        for class_folder in os.listdir(self.root):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                # If current entry is a directory, use its name as the label
                for data_file in os.listdir(class_path):
                    labels[data_file] = class_folder
        return labels

    def load_data_eager(self):
        self.data = {}
        for class_folder in os.listdir(self.root):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                # If current entry is a directory, use its name as the label
                for data_file in os.listdir(class_path):
                    file_path = os.path.join(class_path, data_file)
                    self.data[data_file] = self._load_data(file_path)
        return self.data, self.labels[data_file]

    def load_data_lazy(self):
        self.data = {}
        for class_folder in os.listdir(self.root):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                # If current entry is a directory, use its name as the label
                for data_file in os.listdir(class_path):
                    file_path = os.path.join(class_path, data_file)
                    self.data[data_file] = self._load_data(file_path)
                    yield self.data[data_file], self.labels[data_file]
