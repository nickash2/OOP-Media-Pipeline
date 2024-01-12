class AbstractDataset():
    def __init__(self, root):
        super().__init__(root)
        self.labels = self._load_labels()

    def _load_labels(self):
        raise NotImplementedError  # implement

    def __len__(self):
        # change this to make unlabeled dataset to work as well
        return len(self.labels)
