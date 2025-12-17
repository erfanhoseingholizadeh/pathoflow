import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from typing import List
from .interface import ModelInterface
import logging

logger = logging.getLogger("pathoflow.engine.cnn")

class ResNetClassifier(ModelInterface):
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224), # ResNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load(self, weights_path: str = None):
        logger.info(f"Loading ResNet18 on {self.device}...")
        # We load a pre-trained ResNet18
        # weights='DEFAULT' downloads the ImageNet weights automatically
        self.model = models.resnet18(weights='DEFAULT')
        
        # We are doing Feature Extraction (or binary classification)
        # Let's assume a binary problem: Tumor (1) vs Normal (0)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode (freezes Batch Norm / Dropout)
        logger.info("Model loaded successfully.")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Converts Numpy (H, W, 3) -> Tensor (3, 224, 224)."""
        return self.transform(image)

    def predict_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Runs inference on a batch of images.
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1. Preprocess all images in the batch
        batch_tensors = [self.preprocess(img) for img in images]
        
        # 2. Stack into a single tensor: (Batch, 3, 224, 224)
        batch_input = torch.stack(batch_tensors).to(self.device)

        # 3. Inference (No Gradient calculation = Faster & Less RAM)
        with torch.no_grad():
            outputs = self.model(batch_input)
            
            # Apply Softmax to get probabilities (0 to 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # 4. Return as numpy array (move to CPU first)
        return probabilities.cpu().numpy()