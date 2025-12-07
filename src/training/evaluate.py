import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.data_loader import create_dataloaders
from src.models.model import VisionGuardModel

def evaluate_model():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Evaluation on: {device}")
    
    # Load Data
    _, val_loader = create_dataloaders("configs/config.yaml")
    
    # Load Model
    model = VisionGuardModel(num_classes=2, pretrained=False)
    model_path = "models_saved/dinov2_best.pt"
    
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found. Did you download it?")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # 2. Run Inference Loop
    y_true = []
    y_pred = []
    inference_times = []
    
    print("⏳ Running inference on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Measure Speed
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            # Record time per batch (divide by batch size for single img)
            batch_time = (end_time - start_time) / images.size(0)
            inference_times.append(batch_time)
            
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 3. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    avg_latency = np.mean(inference_times) * 1000 # Convert to ms
    
    print("\n" + "="*30)
    print("  📊 VISIONGUARD EVALUATION REPORT")
    print("="*30)
    print(f"✅ Accuracy:        {accuracy*100:.2f}%")
    print(f"⚡ Avg Latency:     {avg_latency:.2f} ms/image")
    print("-" * 30)
    
    # Detailed Report
    target_names = ['FAKE', 'REAL']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)
    
    # 4. Generate Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('VisionGuard Confusion Matrix')
    
    # Save the plot for your report
    plt.savefig('evaluation_matrix.png')
    print("\n📈 Confusion Matrix saved as 'evaluation_matrix.png'")

if __name__ == "__main__":
    evaluate_model()