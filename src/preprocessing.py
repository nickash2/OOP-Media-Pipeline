from abc import ABC, abstractmethod
import numpy as np


class AbtractPreprocessor(ABC):
    def __init__(self, hyperparameters):
        self.hyperparameters = []

    @abstractmethod
    def __call__(self, text):
        pass

    @abstractmethod
    def _preprocess(self, text):
        pass


class CentreCrop(AbtractPreprocessor):
    def __init__(self, width, height):
        self.width_param = width
        self.height_param = height

    def __call__(self, img):
        self.img = img
        self.img = self._preprocess(img)
        return self.img

    def _preprocess(self, img):
        width, height = img.size
        if self.width_param >= width or self.height_param >= height:
            raise ValueError('Image is smaller than the crop size')

        left = max(0, (width - self.width_param)/2)
        top = max(0, (height - self.height_param)/2)
        right = min(width, left + self.width_param)
        bottom = min(height, top + self.height_param)

        return np.array(img.crop((left, top, right, bottom)).convert('RGB'))
