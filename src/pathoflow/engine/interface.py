from abc import ABC, abstractmethod
import numpy as np
from typing import List

class ModelInterface(ABC):
    """
    The Abstract Base Class (Contract) for all models.
    Any model plugged into PathoFlow MUST implement these methods.
    """

    @abstractmethod
    def load(self, weights_path: str = None):
        """Load weights into memory."""
        pass

    @abstractmethod
    def predict_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Takes a list of RGB numpy images (H, W, C).
        Returns a numpy array of predictions (Batch_Size, Num_Classes).
        """
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Transforms a raw patch into the model's expected format."""
        pass