import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.model import VisionGuardModel

def export_model():
    print("🚀 Starting ONNX Export (Standard Mode)...")
    
    # 1. Load PyTorch Model
    device = torch.device("cpu")
    model = VisionGuardModel(num_classes=2, pretrained=False)
    
    weights_path = "models_saved/dinov2_best.pt"
    if not os.path.exists(weights_path):
        print(f"❌ Error: Could not find {weights_path}")
        return

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("✅ PyTorch Model loaded.")

    # 2. Export to ONNX
    onnx_path = "models_saved/visionguard.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # We use Opset 14 (Stable)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅ Success! ONNX Model saved to: {onnx_path}")
    print(f"📦 Model Size: {file_size:.2f} MB")
    print("👉 You can now update 'app/main.py' to point to this file!")

if __name__ == "__main__":
    export_model()