from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    def __init__(self, root, data_type):
        self._root = root
        self._data_type = data_type
        self._labels = self._load_labels()
        self._data = None

    @abstractmethod
    def _load_labels(self):
        pass

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        self._data_type = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def load_data_eager(self):
        pass

    @abstractmethod
    def load_data_lazy(self):
        pass

    @abstractmethod
    def _load_data(self, file_path):
        pass

    @abstractmethod
    def split(self, ratio):
        pass
