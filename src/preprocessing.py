from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import librosa


class AbtractPreprocessor(ABC):
    """
    Abstract base class for preprocessors.
    """

    def __init__(self, hyperparameters):
        """
        Initialize the preprocessor with hyperparameters.

        Args:
            hyperparameters: A list of hyperparameters.
        """
        self.hyperparameters = []

    @abstractmethod
    def __call__(self, text):
        """
        Apply the preprocessor to the input.

        Args:
            text: The input text.

        Returns:
            The preprocessed text.
        """
        pass

    @abstractmethod
    def _preprocess(self, text):
        """
        Preprocess the input text.

        Args:
            text: The input text.

        Returns:
            The preprocessed text.
        """
        pass


# Images
class CentreCrop(AbtractPreprocessor):
    """
    Preprocessor that performs center cropping on images.
    """

    def __init__(self, width: int, height: int) -> None:
        """
        Initialize the CentreCrop preprocessor.

        Args:
            width: The desired width of the cropped image.
            height: The desired height of the cropped image.

        Raises:
            TypeError: If width or height is not an integer.
            ValueError: If width or height is less than or equal to zero.
        """
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("width and height must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than zero")

        self.width_param = width
        self.height_param = height

    def __call__(self, img: Image) -> Image:
        """
        Apply the CentreCrop preprocessor to the input image.

        Args:
            img: The input image.

        Returns:
            The cropped image.

        Raises:
            TypeError: If img is not an Image (PIL) object.
        """
        if not isinstance(img, Image):
            raise TypeError("img must be an Image (PIL) object")

        self.real_width, self.real_height = img.size
        self.img = self._preprocess(Image.fromarray(img))
        return self.img

    def _check_valid_size(self, img: Image) -> bool:
        """
        Check if the input image has valid size for cropping.

        Args:
            img: The input image.

        Returns:
            True if the image has valid size for cropping, False otherwise.
        """
        if (self.width_param < self.real_width and
                self.height_param < self.real_height):
            return True

    def _preprocess(self, img: Image) -> np.ndarray:
        """
        Perform the center cropping on the input image.

        Args:
            img: The input image.

        Returns:
            The cropped image as a NumPy array.
        """
        if self._check_valid_size(img):
            return np.array(img)

        left = max(0, (self.real_width - self.width_param) / 2)
        top = max(0, (self.real_height - self.height_param) / 2)
        right = min(self.real_width, left + self.width_param)
        bottom = min(self.real_height, top + self.height_param)

        return np.array(img.crop((left, top, right, bottom)))


class RandomCrop(CentreCrop):
    """
    Preprocessor that performs random cropping on images.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize the RandomCrop preprocessor.

        Args:
            width: The desired width of the cropped image.
            height: The desired height of the cropped image.
        """
        super().__init__(width, height)

    def _preprocess(self, img: Image) -> np.ndarray:
        """
        Perform the random cropping on the input image.

        Args:
            img: The input image.

        Returns:
            The cropped image as a NumPy array.
        """
        if self._check_valid_size(img):
            return np.array(img)

        left = np.random.randint(0, self.real_width - self.width_param + 1)
        top = np.random.randint(0, self.real_height - self.height_param + 1)
        right = left + self.width_param
        bottom = top + self.height_param
        return np.array(img.crop((left, top, right, bottom)))


# Audio

# Pitch shift
class PitchShift(AbtractPreprocessor):
    """
    Preprocessor that performs pitch shifting on audio.
    """

    def __init__(self, pitch_factor: float, sample_rate: float):
        """
        Initialize the PitchShift preprocessor.

        Args:
            pitch_factor: The pitch shifting factor.
            sample_rate: The sample rate of the audio.
        """
        self.pitch_factor = pitch_factor
        self.sample_rate = sample_rate

    def __call__(self, audio):
        """
        Apply the PitchShift preprocessor to the input audio.

        Args:
            audio: The input audio.

        Returns:
            The pitch-shifted audio.
        """
        self.audio = self._preprocess(audio)
        return self.audio

    def _preprocess(self, audio):
        """
        Perform the pitch shifting on the input audio.

        Args:
            audio: The input audio.

        Returns:
            The pitch-shifted audio.
        """
        shifted = librosa.effects.pitch_shift(
            y=audio, sr=self.sample_rate, n_steps=self.pitch_factor
        )
        return shifted


# mel spectrogram
class MelSpectrogram(AbtractPreprocessor):
    """
    Preprocessor that computes the mel spectrogram of audio.
    """

    def __init__(self, sample_rate: float, file_name: str = ""):
        """
        Initialize the MelSpectrogram preprocessor.

        Args:
            sample_rate: The sample rate of the audio.
            file_name: The name of the file to save the spectrogram image.
        """
        self.sample_rate = sample_rate
        self.file_name = file_name

    def __call__(self, audio):
        """
        Apply the MelSpectrogram preprocessor to the input audio.

        Args:
            audio: The input audio.

        Returns:
            The mel spectrogram of the audio.
        """
        self.audio = self._preprocess(audio)
        return self.audio

    def _preprocess(self, audio):
        """
        Compute the mel spectrogram of the input audio.

        Args:
            audio: The input audio.

        Returns:
            The mel spectrogram of the audio.
        """
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            librosa.power_to_db(mel, ref=np.max),
            y_axis="mel",
            fmax=8000,
            x_axis="time"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel spectrogram")
        plt.tight_layout()
        if self.file_name:
            plt.savefig(self.file_name)
        return mel


class PreprocessingPipeline(AbtractPreprocessor):
    """
    A pipeline for preprocessing audio data.

    This pipeline applies a series of preprocessing operations to
    an audio signal.
    Each operation is represented by a preprocessor object, and the
    preprocessors are applied in the order they are given in the constructor.

    Note: A `MelSpectrogram` operation cannot be
    followed by a `PitchShift` operation.

    Parameters
    ----------
    *preprocessors : AbtractPreprocessor
        The preprocessors to apply.

    Raises
    ------
    ValueError
        If a `MelSpectrogram` operation is followed
        by a `PitchShift` operation.
    """
    def __init__(self, *preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, audio):
        for i, preprocessor in enumerate(self.preprocessors):
            # Check if a MelSpectrogram is followed by a PitchShift operation
            if (isinstance(preprocessor, MelSpectrogram) and
                i < len(self.preprocessors) - 1 and
                    isinstance(self.preprocessors[i + 1], PitchShift)):
                raise ValueError("A MelSpectrogram operation cannot be" +
                                 "followed by a PitchShift operation.")
            audio = preprocessor(audio)
        return audio
