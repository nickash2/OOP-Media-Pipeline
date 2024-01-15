import os
import sys

sys.path.append(os.getcwd() + "/src/")
from dataset import LabeledDataset 

def main():
    dataset = LabeledDataset(root="root", data_type="image", label_file="labels.csv")
    print(dataset.load_data_eager())


if __name__ == "__main__":
    main()
