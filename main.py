import os
import sys

sys.path.append(os.getcwd() + "/src/")
from dataset import HierarchicalDataset, LabeledDataset


def main():
    hier = HierarchicalDataset(root="data/hierarchical", data_type="image")
    # hier.load_data_eager()
    print(hier[0][1])
    # print(hier.data)


if __name__ == "__main__":
    main()
