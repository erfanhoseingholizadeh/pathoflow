# This training part was done on Google Colab.

# 1. Libraries (Install & Import)
!pip install datasets -q

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from datasets import load_dataset
from tqdm.notebook import tqdm
import numpy as np
from google.colab import drive
import os
import random
from sklearn.metrics import accuracy_score

# --- REPRODUCIBILITY SETUP ---
def set_seed(seed=100):
    """Sets the seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 Seed set to {seed} for reproducibility.")

set_seed(100)
# --- MOUNT GOOGLE DRIVE ---
drive.mount('/content/drive')

save_dir = "/content/drive/MyDrive/PathoFlow_Models"
os.makedirs(save_dir, exist_ok=True)

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 20           
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = os.path.join(save_dir, "pathoflow_resnet18_pro.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Hardware Check: {DEVICE}")

# 2. Load Data via Hugging Face
print("Streaming PCam Dataset from Hugging Face...")
dataset = load_dataset("1aurent/PatchCamelyon", trust_remote_code=True)

# 3. Create Wrapper
class PCamWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert("RGB")
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# 4. Medical Grade Augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Data Loaders
train_ds = PCamWrapper(dataset['train'], transform=train_transform)
val_ds = PCamWrapper(dataset['valid'], transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# 6. Initialize Model
print("Initializing ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1) # Binary Head
model = model.to(DEVICE)

# 7. Optimizer, Loss & AMP Scaler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler()

# 8. Training Loop
best_acc = 0.0

print(f"\nSTARTING TRAINING ({EPOCHS} Epochs)")
print("="*50)

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(train_ds)

    # --- VALIDATE ---
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            labels = labels.float().unsqueeze(1)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float() # Temporary threshold for monitoring

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = correct / total
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    print(f"   RESULTS: Loss: {epoch_loss:.4f} | Accuracy: {val_acc:.4f} | LR: {current_lr:.6f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ NEW BEST MODEL SAVED to Drive ({val_acc:.4f})")

    print("-" * 50)

print(f"\nDONE. Best weights (Acc: {best_acc:.4f}) saved at: {MODEL_SAVE_PATH}")

# ==========================================
# 🔎 HIERARCHICAL THRESHOLD SEARCH
# ==========================================
print("\n🔎 STARTING THRESHOLD OPTIMIZATION...")

# 1. Reload Best Model
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# 2. Get Raw Predictions
y_true = []
y_probs = []

print("   Collecting predictions from validation set...")
with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Inference"):
        inputs = inputs.to(DEVICE)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
        y_probs.extend(probs)
        y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# 3. Hierarchical Search Strategy
def find_best_threshold(start, end, step):
    thresholds = np.arange(start, end, step)
    best_t = 0.5
    best_a = 0.0
    for t in thresholds:
        acc = accuracy_score(y_true, (y_probs > t).astype(int))
        if acc > best_a:
            best_a = acc
            best_t = t
    return best_t, best_a

# --- PHASE 1: Coarse Scan (0.00 to 1.00) ---
final_threshold, final_acc = find_best_threshold(0.00, 1.01, 0.01)
print(f"   Phase 1 (0-100): Best Threshold = {final_threshold:.2f} | Acc = {final_acc:.4f}")

# --- PHASE 2: Zoom in (0.00 to 0.50) if needed ---
if final_threshold < 0.50:
    print("   📉 Threshold is low (< 0.50). Zooming in...")
    final_threshold, final_acc = find_best_threshold(0.00, 0.51, 0.005) # Finer step
    print(f"   Phase 2 (0-50):  Best Threshold = {final_threshold:.3f} | Acc = {final_acc:.4f}")

    # --- PHASE 3: Ultra Zoom (0.00 to 0.10) if needed ---
    if final_threshold < 0.10:
        print("   📉 Threshold is very low (< 0.10). Ultra-Zooming...")
        final_threshold, final_acc = find_best_threshold(0.00, 0.101, 0.001) # Ultra fine step
        print(f"   Phase 3 (0-10):  Best Threshold = {final_threshold:.3f} | Acc = {final_acc:.4f}")

print("="*50)
print(f"🏆 FINAL OPTIMAL THRESHOLD: {final_threshold:.3f}")
print(f"🚀 FINAL OPTIMAL ACCURACY:  {final_acc:.4f}")
print("="*50)

# ==========================================
# 💾 SAVE FINAL PRODUCTION ARTIFACT
# ==========================================
# We overwrite the file one last time to include the threshold metadata
checkpoint = {
    'model_state_dict': model.state_dict(),
    'threshold': final_threshold,
    'accuracy': final_acc,
    'class_names': ['No Tumor', 'Tumor'],
    'architecture': 'resnet18'
}
torch.save(checkpoint, MODEL_SAVE_PATH)
print(f"✅ Production Model with Metadata saved to: {MODEL_SAVE_PATH}")