import os
import librosa
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


class AbstractDataset:
    def __init__(self, root, data_type):
        self.root = root
        self.data_type = data_type
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
        self.data = self._load_data()
        return self.data

    def load_data_lazy(self, idx):
        if self.data is None:
            self.data = {}
        if idx not in self.data:
            self.data[idx] = self._load_data(self.data_paths[idx])
        else:
            raise ValueError("Data already loaded")

    def _load_data(self):
        data = {}
        for class_folder in os.listdir(self.root):
            class_path = os.path.join(self.root, class_folder)
            if os.path.isdir(class_path):
                for data_file in os.listdir(class_path):
                    file_path = os.path.join(class_path, data_file)
                    if self.data_type == 'image':
                        img = Image.open(file_path)
                        data[data_file] = np.array(img.convert("RGB"))
                    elif self.data_type == 'audio':
                        audio, sr = librosa.load(file_path)
                        data[data_file] = (audio, sr)
        return data

    def split(self, ratio):
        # split me
        files = list(self.data.keys())
        train_files, test_files = train_test_split(files, test_size=ratio)
        train_data = {file: self.data[file] for file in train_files}
        test_data = {file: self.data[file] for file in test_files}
        return train_data, test_data
