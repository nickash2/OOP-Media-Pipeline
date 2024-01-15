import os

from sklearn.model_selection import train_test_split


class AbstractDataset:
    def __init__(self, root):
        super().__init__(root)
        self.labels = self._load_labels()
        self.data = None
        self.data_paths = [os.path.join(root, file)
                           for file in os.listdir(root)]

    def _load_labels(self):
        raise NotImplementedError  # implement

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.data is None:
            raise ValueError("Data is not loaded")
        if idx not in self.data:
            self.load_data_lazy(idx)
        return self.data[idx]

    def load_data_eager(self):
        self.data = {file: self._load_data(file)
                     for file in os.listdir(self.root)}

    def load_data_lazy(self, idx):
        if self.data is None:
            self.data = {}
        if idx not in self.data:
            self.data[idx] = self._load_data(self.data_paths[idx])

    def _load_data(self, filename):
        raise NotImplementedError

    def split(self, ratio):
        # split me
        files = list(self.data.keys())
        train_files, test_files = train_test_split(files, test_size=ratio)
        train_data = {file: self.data[file] for file in train_files}
        test_data = {file: self.data[file] for file in test_files}
        return train_data, test_data
