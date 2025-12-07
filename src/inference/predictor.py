import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

# Import GradCAM tools
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models.model import VisionGuardModel

class VisionGuardPredictor:
    def __init__(self, model_path, config_path="configs/config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading Inference Engine on: {self.device}")
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
        
        try:
            # 1. Load Model
            self.model = VisionGuardModel(num_classes=2, pretrained=False)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading model: {str(e)}")
        
        try:
            # 2. Setup GradCAM (The Explainability Tool)
            # We target the last normalization layer of the backbone
            target_layers = [self.model.backbone.blocks[-1].norm1]
            
            # DINOv2 requires a special reshape transform because it outputs 1D sequences
            def reshape_transform(tensor):
                # DINOv2 small outputs: [Batch, 257, 384] (1 CLS token + 256 Patches)
                # We discard the CLS token (index 0) and keep the 256 patches
                result = tensor[:, 1:, :]
                
                # Reshape 256 -> 16x16 grid (since 224/14 = 16)
                height = 14
                width = 14
                # Note: If image size is 224x224, grid is 16x16. 
                # DINOv2-S/14 means patch size is 14. 224/14 = 16.
                grid_size = 16
                
                result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
                
                # Bring channels first: [Batch, Channels, Height, Width]
                result = result.transpose(2, 3).transpose(1, 2)
                return result

            self.cam = GradCAM(model=self.model, target_layers=target_layers, reshape_transform=reshape_transform)
            print("✅ GradCAM initialized successfully")
        except Exception as e:
            print(f"⚠️  Warning: GradCAM initialization failed: {str(e)}")
            self.cam = None

        # 3. Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.labels = ['FAKE', 'REAL']

    def predict(self, image_path):
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image file not found: {image_path}")
        
        try:
            # 1. Load Image
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"❌ Error loading image: {str(e)}")
        
        try:
            # Keep a clean copy for visualization (resized to 224x224)
            vis_image = image.resize((224, 224))
            vis_image = np.float32(vis_image) / 255.0 # Normalize 0-1 for OpenCV
            
            # 2. Transform for Model
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 3. Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)

            # 4. Generate Heatmap
            heatmap_pil = None
            if self.cam is not None:
                try:
                    # We tell GradCAM to look for the predicted class
                    grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
                    grayscale_cam = grayscale_cam[0, :]
                    
                    # Overlay heatmap on image
                    visualization = show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)
                    
                    # Convert back to PIL for Gradio
                    heatmap_pil = Image.fromarray(visualization)
                except Exception as e:
                    print(f"⚠️  Warning: Heatmap generation failed: {str(e)}")
            
            if heatmap_pil is None:
                # Fallback: return original image if heatmap fails
                heatmap_pil = image.resize((224, 224))

            # 5. Format Output
            idx = predicted_class.item()
            return {
                "verdict": self.labels[idx],
                "confidence": round(float(confidence.item()) * 100, 2),
                "probabilities": {
                    "FAKE": round(float(probs[0][0].item()), 4),
                    "REAL": round(float(probs[0][1].item()), 4)
                },
                "heatmap": heatmap_pil
            }
        except Exception as e:
            raise RuntimeError(f"❌ Error during prediction: {str(e)}")