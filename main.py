import os
import sys

from PIL import Image

sys.path.append(os.getcwd() + "/src/")
from dataset import LabeledDataset


def main():
    age_dataset = LabeledDataset(
        root="datasets/age-estimation-classification",
        label_file="train.csv",
        data_type="image",
    )
    age_dataset.load_data_eager()
    img, label = age_dataset[2]

    print(Image.fromarray(img).size)
    print(label)


if __name__ == "__main__":
    main()
