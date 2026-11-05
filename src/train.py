import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.config import cfg
from src.dataset import SignSkeletonDataset, generate_synthetic_dataset
from src.models.sign_classifier import STGCNSignClassifier

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[START] Running training pipeline on device: {device}")

    x_path = cfg.DATA_DIR / "X.npy"
    y_path = cfg.DATA_DIR / "y.npy"
    if x_path.exists() and y_path.exists():
        print(f"Loading processed data from {cfg.DATA_DIR}...")
        X, y = np.load(x_path), np.load(y_path)
    else:
        print("Processed dataset not found. Generating 400 synthetic benchmark sequences...")
        X, y = generate_synthetic_dataset(num_samples=400)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_loader = DataLoader(
        SignSkeletonDataset(X_train, y_train, augment=True),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        SignSkeletonDataset(X_val, y_val, augment=False),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False
    )

    model = STGCNSignClassifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    best_val_acc = 0.0
    best_ckpt_path = cfg.CHECKPOINT_DIR / "best_stgcn.pt"

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_x.size(0)

        scheduler.step()
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_logits = model(val_x)
                val_correct += (val_logits.argmax(dim=1) == val_y).sum().item()
                val_total += val_x.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{cfg.EPOCHS:02d}] Loss: {train_loss / total:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)

    print(f"[DONE] Training complete. Best Accuracy: {best_val_acc:.2%} saved to {best_ckpt_path}")

if __name__ == "__main__":
    train_model()