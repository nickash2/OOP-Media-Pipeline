import os
import sys

sys.path.append(os.getcwd() + "/src/")
from dataset import UnlabeledDataset, HierarchicalDataset, LabeledDataset


def main():
    labeled = LabeledDataset(root="data/labeled/",label_file="labels.csv", data_type="image")
    # miu = HierarchicalLabeledDataset(root="data/hierarchical/", data_type="image")
    # miu.load_data_eager()
    print(labeled[5])
    


if __name__ == "__main__":
    main()
