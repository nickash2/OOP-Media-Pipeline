from abstract_dataset import AbstractDataset


class LabeledDataset(AbstractDataset):
    def __init__(self, root):
        super().__init__(root)

    def __len__(self):
        # Implement the logic to return the number of datapoints in the labeled dataset
        pass


class UnlabeledDataset(AbstractDataset):
    def __init__(self, root):
        super().__init__(root)

    def __len__(self):
        # Implement the logic to return the number of datapoints in the unlabeled dataset
        pass
