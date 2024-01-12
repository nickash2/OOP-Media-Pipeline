class AbstractDataset:
    def __init__(self, root):
        self.root = root

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__ method")

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__ method")
