import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathoflow.engine.interface import ModelInterface
from pathlib import Path

class ResNetClassifier(ModelInterface):
    def __init__(self, model_path: Path = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.threshold = 0.5  # Default fallback
        
        # Define standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Call the mandatory load method immediately
        self.model = self.load(model_path)
        self.model.eval()

    def load(self, model_path: Path):
        """
        Loads the model and applies 'torch.compile' optimization.
        """
        # 1. Define Architecture
        model = models.resnet18(weights=None) 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1) # Binary Head
        model = model.to(self.device)

        # 2. Load Weights & Metadata
        if model_path and Path(model_path).exists():
            print(f"   [AI] 🧠 Loading Brain from: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # --- Handle Production File ---
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    # It's the new Production file!
                    model.load_state_dict(checkpoint["model_state_dict"])
                    self.threshold = checkpoint.get("threshold", 0.5)
                    acc = checkpoint.get("accuracy", 0.0)
                    print(f"   [AI] ✅ Production Mode. Threshold: {self.threshold} | Accuracy: {acc:.2%}")
                else:
                    # It's a standard .pth file (Old)
                    model.load_state_dict(checkpoint)
                    print("   [AI] ✅ Standard Weights loaded.")

            except Exception as e:
                print(f"   [AI] ❌ Error loading weights: {e}")
                print("   [AI] ⚠️ Falling back to Random Weights.")
        else:
            print("   [AI] ⚠️ No model path found. Using Random Weights.")

        # --- SMART COMPILATION ---
        # ONLY compile if we are on a GPU (CUDA) to avoid CPU/WSL warnings.
        if self.device.type == 'cuda' and hasattr(torch, "compile"):
            print("   [AI] 🚀 GPU Detected: Compiling model with mode='max-autotune'...")
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                print(f"   [AI] ⚠️ Compilation failed (running standard mode): {e}")
        else:
            print("   [AI] ℹ️  Standard Inference Mode (Compilation skipped for stability)")

        return model

    def preprocess(self, image: Image.Image):
        """
        Mandatory implementation of the preprocess method.
        """
        return self.transform(image)

    def predict_batch(self, patches: list[Image.Image]) -> np.ndarray:
        if not patches:
            return np.array([])
            
        # Use the mandatory preprocess method
        tensors = [self.preprocess(p) for p in patches]
        batch_tensor = torch.stack(tensors).to(self.device)

        # --- INFERENCE MODE ---
        # 'inference_mode' is faster and uses less memory than 'no_grad'
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.sigmoid(logits)
            
        return probs.cpu().numpy().flatten()