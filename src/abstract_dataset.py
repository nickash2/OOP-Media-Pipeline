from abc import ABC, abstractmethod


class AbstractDataset(ABC):
    def __init__(self, root, data_type):
        self.root = root
        self.data_type = data_type
        self.labels = self._load_labels()
        self.data = None

    @abstractmethod
    def _load_labels(self):
        pass

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
