import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import os

# Import our project modules
from src.data.data_loader import create_dataloaders
from src.models.model import VisionGuardModel

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    return running_loss / len(loader), 100 * correct / total

def main():
    # 1. Config & Device
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Determine device
    device_cfg = cfg['model'].get('device', 'auto')
    if device_cfg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    print(f"🚀 Training on: {device}")

    # 2. Setup Save Directory
    save_dir = cfg['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dinov2_best.pt")

    # 3. Load Data & Model
    try:
        train_loader, val_loader = create_dataloaders("configs/config.yaml")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    model = VisionGuardModel(num_classes=2).to(device)

    # 4. Optimizer (Only training the Head)
    # Note: We only pass model.head.parameters() to optimizer because backbone is frozen!
    optimizer = optim.AdamW(model.head.parameters(), lr=float(cfg['model']['learning_rate_head']))
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    epochs = cfg['model']['epochs']
    patience = cfg['training'].get('patience', 3)
    patience_counter = 0
    
    print(f"\n🔥 Starting Training for {epochs} Epochs...")
    print(f"💾 Best model will be saved to: {save_path}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   ⭐ Saved New Best Model ({best_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered (no improvement for {patience} epochs)")
                break
    
    print(f"\n✅ Training Complete! Best model saved to: {save_path}")
    return save_path

if __name__ == "__main__":
    main()