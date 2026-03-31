"""
Train a cat identity classifier using the labeled images.

Usage:
    python train_classifier.py

Extracts cat ROIs from the labeled folders using the green bounding box,
fine-tunes a pretrained MobileNetV2 to classify bonnie / jinny / louise,
and saves the model to cat_classifier.pt.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

BASE_DIR   = "cat images"
CATS       = ["bonnie", "jinny", "louise"]
OUTPUT     = "cat_classifier.pt"
IMG_SIZE   = 224
EPOCHS     = 60
BATCH_SIZE = 8
LR         = 1e-3

BOX_HSV_LOWER = np.array([42, 150, 150])
BOX_HSV_UPPER = np.array([62, 255, 255])


# ---------------------------------------------------------------------------
# ROI extraction (same logic as tune_colors.py)
# ---------------------------------------------------------------------------

def find_box_roi(img: np.ndarray) -> np.ndarray | None:
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BOX_HSV_LOWER, BOX_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask   = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 20 or h < 20:
        return None
    pad = 4
    roi = img[y + pad: y + h - pad, x + pad: x + w - pad]
    return roi if roi.size > 0 else None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CatDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, int]], transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        roi, label = self.samples[idx]
        img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        return self.transform(img), label


def load_samples() -> list[tuple[np.ndarray, int]]:
    samples = []
    for label, cat in enumerate(CATS):
        folder = os.path.join(BASE_DIR, cat)
        files  = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        found  = 0
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            roi = find_box_roi(img)
            if roi is None:
                continue
            samples.append((roi, label))
            found += 1
        print(f"  {cat}: {found} ROIs extracted")
    return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print("Extracting ROIs...")
    samples = load_samples()
    if not samples:
        print("No samples found — check your image folders.")
        return

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = CatDataset(samples, train_tf)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    print(f"\nTraining on {len(samples)} samples ({len(CATS)} classes)...")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(CATS))

    # Freeze the feature extractor, only train the classifier head
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss, correct, n = 0.0, 0, 0
        for imgs, labels in loader:
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            n          += len(labels)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{EPOCHS}  loss={total_loss/n:.4f}  acc={correct/n:.0%}")

    # Unfreeze and fine-tune the whole network at a lower LR
    print("\nFine-tuning full network...")
    for p in model.features.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)

    for epoch in range(1, 21):
        total_loss, correct, n = 0.0, 0, 0
        for imgs, labels in loader:
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            n          += len(labels)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/20   loss={total_loss/n:.4f}  acc={correct/n:.0%}")

    torch.save({"state_dict": model.state_dict(), "classes": CATS}, OUTPUT)
    print(f"\nSaved to {OUTPUT}")

    # Quick sanity check
    print("\nSanity check (training set):")
    model.eval()
    infer_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    correct, n = 0, 0
    with torch.no_grad():
        for roi, label in samples:
            img  = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            out  = model(infer_tf(img).unsqueeze(0))
            pred = out.argmax(1).item()
            correct += pred == label
            n += 1
    print(f"  {correct}/{n} correct ({correct/n:.0%})")


if __name__ == "__main__":
    train()
