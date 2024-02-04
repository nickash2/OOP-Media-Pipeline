from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import librosa
from typing import List, Union


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
        self._hyperparameters = []

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

        self._width_param = width
        self._height_param = height

    @property
    def width_param(self):
        return self._width_param

    @width_param.setter
    def width_param(self, value):
        if not isinstance(value, int):
            raise TypeError("width_param must be an integer")
        if value <= 0:
            raise ValueError("width_param must be greater than zero")
        self._width_param = value

    @property
    def height_param(self):
        return self._height_param

    @height_param.setter
    def height_param(self, value):
        if not isinstance(value, int):
            raise TypeError("height_param must be an integer")
        if value <= 0:
            raise ValueError("height_param must be greater than zero")
        self._height_param = value

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply the CentreCrop preprocessor to the input image.

        Args:
            img: The input image.

        Returns:
            The cropped image.

        Raises:
            TypeError: If img is not an Image (PIL) object.
        """
        if not isinstance(img, Image.Image) and isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8'))

        elif not isinstance(img, Image.Image):
            raise TypeError("img must be an Image (PIL) object")

        self._real_width, self._real_height = img.size
        self.img = self._preprocess(img)
        return self.img

    def _check_valid_size(self, img: Image) -> bool:
        """
        Check if the input image has valid size for cropping.

        Args:
            img: The input image.

        Returns:
            True if the image has valid size for cropping, False otherwise.
        """
        if (self._width_param < self._real_width and
                self._height_param < self._real_height):
            return True

    def _preprocess(self, img: Image) -> np.ndarray:
        """
        Perform the center cropping on the input image.

        Args:
            img: The input image.

        Returns:
            The cropped image as a NumPy array.
        """
        if not self._check_valid_size(img):
            return np.array(img)

        left = max(0, (self._real_width - self._width_param) / 2)
        top = max(0, (self._real_height - self._height_param) / 2)
        right = min(self._real_width, left + self._width_param)
        bottom = min(self._real_height, top + self._height_param)

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
        if not self._check_valid_size(img):
            return np.array(img)

        left = np.random.randint(0, self._real_width - self._width_param + 1)
        top = np.random.randint(0, self._real_height - self._height_param + 1)
        right = left + self._width_param
        bottom = top + self._height_param
        return np.array(img.crop((left, top, right, bottom)))


# Audio
class PitchShift(AbtractPreprocessor):
    """
    Preprocessor that performs pitch shifting on audio.
    """

    def __init__(self, pitch_factor: float, sample_rate: float):
        """
        Initialize the PitchShift preprocessor.

        Args:
            _pitch_factor: The pitch shifting factor.
            sample_rate: The sample rate of the audio.
        """
        self._pitch_factor = pitch_factor
        self._sample_rate = sample_rate

    @property
    def pitch_factor(self):
        return self._pitch_factor

    @pitch_factor.setter
    def pitch_factor(self, value):
        self._pitch_factor = value

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the PitchShift preprocessor to the input audio.

        Args:
            audio: The input audio.

        Returns:
            The pitch-shifted audio.
        """
        self.audio = self._preprocess(audio)
        return self.audio

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Perform the pitch shifting on the input audio.

        Args:
            audio: The input audio.

        Returns:
            The pitch-shifted audio.
        """
        shifted = librosa.effects.pitch_shift(
            y=audio, sr=self._sample_rate, n_steps=self._pitch_factor
        )
        return shifted


class MelSpectrogram(AbtractPreprocessor):
    """
    Preprocessor that computes the mel spectrogram of audio.
    """

    def __init__(self, sample_rate: float, file_name: str = "") -> None:
        """
        Initialize the MelSpectrogram preprocessor.

        Args:
            sample_rate: The sample rate of the audio.
            file_name: The name of the file to save the spectrogram image.
        """
        self._sample_rate = sample_rate
        self._file_name = file_name

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        self._sample_rate = value

    @property
    def file_name(self) -> str:
        return self._file_name

    @file_name.setter
    def file_name(self, value: str) -> None:
        self._file_name = value

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the MelSpectrogram preprocessor to the input audio.

        Args:
            audio: The input audio.

        Returns:
            The mel spectrogram of the audio.
        """
        self.audio = self._preprocess(audio)
        return self.audio

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute the mel spectrogram of the input audio.

        Args:
            audio: The input audio.

        Returns:
            The mel spectrogram of the audio.
        """
        mel = librosa.feature.melspectrogram(y=audio, sr=self._sample_rate)
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
        if self._file_name:
            plt.savefig(self._file_name)
        plt.close()
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

    def __init__(self, preprocessors: List[AbtractPreprocessor]):
        """
        Initialize the PreprocessingPipeline.

        Parameters
        ----------
        preprocessors : List[AbtractPreprocessor]
            The preprocessors to apply.
        """
        self._preprocessors = preprocessors
        self._data = None

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessing pipeline to the input data.

        Parameters
        ----------
        data : np.ndarray
            The input audio data.

        Returns:
            np.ndarray: The preprocessed audio data.
        """
        self._data = self._preprocess(data)
        return self._data

    @property
    def preprocessors(self) -> List[AbtractPreprocessor]:
        return self._preprocessors

    @preprocessors.setter
    def preprocessors(self, value: List[AbtractPreprocessor]) -> None:
        self._preprocessors = value

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the preprocessors in the pipeline to the input data.

        Parameters
        ----------
        data : np.ndarray
            The input audio data.

        Returns: np.ndarray The preprocessed audio data.
        """
        for i, preprocessor in enumerate(self._preprocessors):
            # Check if a MelSpectrogram is followed by a PitchShift operation
            if (
                isinstance(preprocessor, MelSpectrogram)
                and i < len(self._preprocessors) - 1
                and isinstance(self._preprocessors[i + 1], PitchShift)
            ):
                raise ValueError(
                    "A MelSpectrogram operation cannot be"
                    + "followed by a PitchShift operation."
                )
            data = preprocessor(data)
        return data
