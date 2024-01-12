from abstract_dataset import AbstractDataset
import os
import csv
from typing import Dict


class LabeledDataset(AbstractDataset):
    """
    A dataset where each data point has a label.
    The labels are loaded from a separate CSV file.
    """

    def __init__(self, root: str, label_file: str) -> None:
        """
        Initialize a LabeledDataset.

        Parameters:
        root (str): The path to the directory containing the data files.
        label_file (str): The path to the CSV file containing the labels.
        """
        self.label_file = label_file
        super().__init__(root)

    def _load_labels(self) -> Dict[str, str]:
        """
        Load labels from the CSV file.
        Each line in the file should be formatted as '<filename>,<label>'.

        Returns:
        dict: A dictionary mapping filenames to labels.
        """
        with open(self.label_file, 'r') as f:
            reader = csv.reader(f)
            # Dictionary where the keys are filenames and the values are labels
            labels = {rows[0]: rows[1] for rows in reader}
        return labels


class UnlabeledDataset(AbstractDataset):
    """
    A dataset where each data point does not have a label.
    """

    def _load_labels(self) -> Dict[str, None]:
        """
        Since the dataset is unlabeled,
        assign None as the label for each data file.

        Returns:
        dict: A dictionary mapping filenames to None.
        """
        return {file: None for file in os.listdir(self.root)}


class HierarchicalLabeledDataset(AbstractDataset):
    """
    A dataset where each data point has a label.
    The labels are determined based on the directory structure.
    Each subdirectory in the root directory represents a class,
    and the files in each subdirectory are the data points for that class.
    """

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
