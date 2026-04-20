import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import DotToCharDataset, transform
import os

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4  #set fixed for all trainings, could be set larger when more data is used, or smaller wehn less data is used
BATCH_SIZE = 8
NUM_EPOCHS = 15   # as a max, should converge faster, following the loss, and avoid overtraining, no automatic early stopping foreseen
SAVE_DIR = "checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice_score = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice_score

bce_criterion = nn.BCEWithLogitsLoss()
dice_criterion = DiceLoss()

# using grayscale
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_dataset = DotToCharDataset("data/train_dots", "data/train_full", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Starting training on {DEVICE}...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for i, (dots, fulls) in enumerate(train_loader):
        dots, fulls = dots.to(DEVICE), fulls.to(DEVICE)
        optimizer.zero_grad()
        predictions = model(dots)
        bce_loss = bce_criterion(predictions, fulls)
        dice_loss = dice_criterion(torch.sigmoid(predictions), fulls)
        loss = 0.5 * bce_loss + 0.5 * dice_loss # used because images contain only few colored pixels compared to white, Otherwise loss goes to 0 very fast and the optimal solution is a white image.
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f} | BCE: {bce_loss.item():.4f} | Dice: {dice_loss.item():.4f} | Pred Mean: {torch.sigmoid(predictions).mean():.4f}")
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
       torch.save(model.state_dict(), f"{SAVE_DIR}/model_epoch_{epoch + 1}_dice.pth")
    avg_loss = running_loss / len(train_loader)

print("Training complete!")