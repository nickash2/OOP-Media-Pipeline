from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class AbtractPreprocessor(ABC):
    def __init__(self, hyperparameters):
        self.hyperparameters = []

    @abstractmethod
    def __call__(self, text):
        pass

    @abstractmethod
    def _preprocess(self, text):
        pass


# Images
class CentreCrop(AbtractPreprocessor):
    def __init__(self, width: int, height: int) -> None:
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("width and height must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than zero")

        self.width_param = width
        self.height_param = height

    def __call__(self, img: Image) -> Image:
        if not isinstance(img, Image):
            raise TypeError("img must be an Image (PIL) object")

        self.real_width, self.real_height = img.size
        self.img = self._preprocess(Image.fromarray(img))
        return self.img

    def _check_valid_size(self, img: Image) -> bool:
        if (self.width_param < self.real_width and
                self.height_param < self.real_height):
            return True

    def _preprocess(self, img: Image) -> np.ndarray:
        if self._check_valid_size(img):
            return np.array(img)

        left = max(0, (self.real_width - self.width_param) / 2)
        top = max(0, (self.real_height - self.height_param) / 2)
        right = min(self.real_width, left + self.width_param)
        bottom = min(self.real_height, top + self.height_param)

        return np.array(img.crop((left, top, right, bottom)))


class RandomCrop(CentreCrop):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    def _preprocess(self, img: Image) -> np.ndarray:
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
    def __init__(self, hyperparameters):
        pass

    def __call__(self, audio):
        self.audio = self._preprocess(audio)
        return self.audio

    def _preprocess(self, audio):
        pass


# mel spectrogram
class MelSpectrogram(AbtractPreprocessor):
    def __init__(self, hyperparameters):
        pass

    def __call__(self, audio):
        self.audio = self._preprocess(audio)
        return self.audio

    def _preprocess(self, audio):
        pass
