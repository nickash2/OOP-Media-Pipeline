import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from typing import List

sys.path.append(os.getcwd() + "/src/")
from dataset import LabeledDataset, UnlabeledDataset
from batchloader import BatchLoader
from preprocessing import (
    PreprocessingPipeline,
    CentreCrop,
    RandomCrop,
    PitchShift,
    MelSpectrogram,
)


def batchloader_discarding(data: np.ndarray, discard: bool) -> None:
    batch_loader = BatchLoader(
        data=data,
        batch_size=32,
        shuffle=False,
        discard_last=discard,
    )

    print(len(batch_loader))
    for i, batch in enumerate(batch_loader):
        print(f"Batch {i} size: {len(batch)}")


def compare_batches(load1, load2):
    for (i, batch1), (j, batch2) in zip(enumerate(load1), enumerate(load2)):
        print(f"First element of batch {i} when shuffle=True: {batch1[0]}")
        print(f"First element of batch {j} when shuffle=False: {batch2[0]}")
        print("---")


def is_batch_shuffling(data):
    batch_loader_shuffle = BatchLoader(
        data=data,
        batch_size=32,
        shuffle=True,
        discard_last=False,
    )

    batch_loader = BatchLoader(
        data=data,
        batch_size=32,
        shuffle=False,
        discard_last=False,
    )
    compare_batches(batch_loader, batch_loader_shuffle)


def save_imgs(batch, batch_index, filenames):
    # Calculate the number of rows and columns for the subplot
    num_images = len(batch)
    num_cols = int(num_images**0.5)
    num_rows = num_images // num_cols + (num_images % num_cols > 0)

    # Create a new figure
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5 * num_cols, 5 * num_rows)
        )

    # Flatten the axes array and remove unused subplots
    axes = axes.flatten()
    for ax in axes[num_images:]:
        fig.delaxes(ax)

    # Plot each image in its own subplot
    for i, (img, ax) in enumerate(zip(batch, axes)):
        ax.imshow(img)

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig(filenames)
    plt.close()


def save_spectrograms(spectrograms, filenames, sr, idx: int = None):
    # Calculate the number of rows and columns for the subplots
    num_images = len(spectrograms)
    num_cols = int(num_images**0.5)
    num_rows = num_images // num_cols + (num_images % num_cols > 0)

    # Create a new figure
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(10 * num_cols, 4 * num_rows)
        )

    # Flatten the axs array
    axs = axs.flatten()

    # Create a subplot for each spectrogram
    for mel, ax in zip(spectrograms, axs):
        spec = librosa.display.specshow(
            librosa.power_to_db(mel, ref=np.max),
            y_axis="mel",
            fmax=8000,
            x_axis="time",
            sr=np.mean(sr),
            ax=ax,
        )
        fig.colorbar(spec, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel spectrogram")

    # Remove the unused subplots
    for ax in axs[num_images:]:
        fig.delaxes(ax)
    filenames = filenames.replace(".png", f"_{idx}.png")
    # Save the entire figure as a single image
    plt.tight_layout()
    plt.savefig(filenames)
    plt.close(fig)


def pipeline_example_batches(
    preprocessors: List[PreprocessingPipeline],
    data: np.ndarray,
    sr=None,
    filenames=None,
    file_type: str = "audio",
    batch_size: int = 16,
) -> None:
    batchloader = BatchLoader(
        data=data,
        batch_size=batch_size,
        shuffle=False,
        discard_last=False,
    )
    print("Number of batches:", len(batchloader))
    pipeline = PreprocessingPipeline(preprocessors)
    for i, batch in enumerate(batchloader):
        # Apply the pipeline to each array in the batch
        print("Processing batch:", i)
        preprocessed_batch = [pipeline(cur_batch) for cur_batch in batch]
        if file_type == "audio":
            save_spectrograms(
                preprocessed_batch,
                filenames=filenames,
                idx=i, sr=sr)
        else:
            save_imgs(preprocessed_batch, i, filenames)


def pipeline_example_dataset(
    preprocessors: List[PreprocessingPipeline],
    data: np.ndarray,
    sr=None,
    filenames=None,
    file_type: str = "audio",
    num_points: int = 50,
) -> None:
    pipeline = PreprocessingPipeline(preprocessors)
    newdata = []
    for i in range(num_points):
        newdata.append(pipeline(data[i]))
    if file_type == "audio":
        save_spectrograms(newdata, filenames=filenames, sr=sr)
    else:
        save_imgs(newdata, i, filenames=filenames)


def lazy_loading_example(age_dataset) -> None:
    for i, (data, label) in enumerate(age_dataset):
        print(f"Loaded {i} element")
        newdata = data[i]
    return newdata


def main() -> None:
    age_dataset = LabeledDataset(
        root="datasets/age-estimation-classification",
        label_file="train.csv",
        data_type="image",
    )

    # Lazy loading the data
    print("Lazy loading example:")
    lazy_loading_example(age_dataset)

    # Egearly loading the data
    print("Eager loading example:")
    age_data, labels = age_dataset.load_data_eager()

    digits_dataset = UnlabeledDataset(
        root="datasets/regression",
        data_type="audio",
    )

    digits_data, sampling_rates = digits_dataset.load_data_eager()

    # # Comment these as you wish to demonstrate discarding
    print("Checking batches with and without discarding")
    batchloader_discarding(age_data, discard=True)
    batchloader_discarding(age_data, discard=False)

    # Compares the first element of each batch
    # when shuffle=True and shuffle=False
    print("Comparing batches with and without shuffling")
    is_batch_shuffling(digits_data)

    # Example of a preprocessing pipeline for audio
    print("Example of a preprocessing pipeline for audio")
    pipeline_example_batches(
        data=digits_data,
        sr=sampling_rates,
        filenames="img/batch_plots/batch_plot01.png",  # Save the plots
        batch_size=64,
        preprocessors=[
            PitchShift(pitch_factor=-5.0, sample_rate=np.mean(sampling_rates)),
            MelSpectrogram(sample_rate=np.mean(sampling_rates)),
        ],
    )

    # # Example of a preprocessing pipeline for imgs
    print("Example of a preprocessing pipeline for imgs")

    pipeline_example_batches(
        data=age_data,
        file_type="img",
        filenames="img/batch_plots/batch_plot02.png",
        preprocessors=[
            CentreCrop(width=80, height=80),
            RandomCrop(width=40, height=40),
        ],
    )

    # Example of a preprocessing pipeline for datasets
    print("Example of a preprocessing pipeline for datasets - imgs")

    pipeline_example_dataset(
        preprocessors=[
            CentreCrop(width=80, height=80),
            RandomCrop(width=40, height=40),
        ],
        data=age_data,
        file_type="img",
        filenames="img/dataset_plots/dataset_plot01.png",
    )

    print("Example of a preprocessing pipeline for datasets - audio")
    pipeline_example_dataset(
        preprocessors=[
            PitchShift(pitch_factor=15, sample_rate=np.mean(sampling_rates)),
            MelSpectrogram(sample_rate=np.mean(sampling_rates)),
        ],
        data=digits_data,
        sr=np.mean(sampling_rates),
        file_type="audio",
        num_points=20,
        filenames="img/dataset_plots/dataset_mel.png",
    )


if __name__ == "__main__":
    main()
