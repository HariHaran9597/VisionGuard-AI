import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import yaml
import os

def get_transforms(cfg):
    """
    DINOv2 expects ImageNet normalization.
    We also add some light augmentation to prevent overfitting.
    """
    img_size = cfg['data']['image_size']
    
    # Training Transforms (with Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Slight color changes
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # DINOv2 Expected Mean
            std=[0.229, 0.224, 0.225]   # DINOv2 Expected Std
        )
    ])

    # Validation/Test Transforms (No Augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def create_dataloaders(config_path="configs/config.yaml"):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    train_transform, val_transform = get_transforms(cfg)
    data_dir = cfg['data']['train_dir'] # Should be "data/raw"

    # Validate data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Data directory not found: {data_dir}")
    
    # Check for required subdirectories
    real_dir = os.path.join(data_dir, "REAL")
    fake_dir = os.path.join(data_dir, "FAKE")
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError(
            f"❌ Expected directory structure not found.\n"
            f"   Expected: {data_dir}/REAL and {data_dir}/FAKE\n"
            f"   Real dir exists: {os.path.exists(real_dir)}\n"
            f"   Fake dir exists: {os.path.exists(fake_dir)}"
        )

    # 1. Load the Entire Dataset (REAL + FAKE)
    try:
        full_dataset = datasets.ImageFolder(root=data_dir)
    except Exception as e:
        raise RuntimeError(f"❌ Error loading dataset from {data_dir}: {str(e)}")
    
    if len(full_dataset) == 0:
        raise ValueError(f"❌ No images found in {data_dir}")
    
    # 2. Split: 80% Train, 20% Validation
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply specific transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 3. Create Loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['data']['num_workers']
    )

    print(f"✅ Data Ready:")
    print(f"   - Train: {len(train_dataset)} images")
    print(f"   - Val:   {len(val_dataset)} images")
    print(f"   - Classes: {full_dataset.class_to_idx}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    create_dataloaders()