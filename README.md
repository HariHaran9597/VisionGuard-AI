# 🛡️ VisionGuard AI

> **Detect AI-Generated Images with Deep Learning & Explainability**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/WebUI-Gradio-orange.svg)](https://gradio.app/)

VisionGuard AI is a state-of-the-art deep learning application that detects synthetic/AI-generated images from real photographs. Built on **Meta's DINOv2** vision foundation model with integrated **GradCAM** explainability, it provides both accurate predictions and interpretable heatmaps showing which image regions trigger AI detection.

---

## 🎯 Features

- ✅ **High-Accuracy Detection** - Leverages DINOv2 (self-supervised vision model) for robust real/fake image classification
- 🔍 **Explainability Built-In** - GradCAM heatmaps reveal exactly which regions trigger AI detection
- ⚡ **Fast Inference** - Optimized for CPU/GPU with <1s per-image latency
- 🎨 **User-Friendly Web UI** - Interactive Gradio interface for non-technical users
- 📊 **Comprehensive Evaluation** - Classification metrics, confusion matrices, and latency analysis
- 🔧 **Production-Ready** - Error handling, validation, and configurable parameters
- 📦 **ONNX Export** - Convert to ONNX format for cross-platform deployment and faster inference
- 📱 **Easy Deployment** - Docker support and cloud-ready architecture

---

## 🚀 Quick Start

### 1. **Clone & Setup**

```bash
git clone https://github.com/yourusername/VisionGuardAI.git
cd VisionGuardAI

# Create virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Prepare Data**

Organize your dataset in the following structure:

```
data/raw/
├── REAL/           # Real photographs
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── FAKE/           # AI-generated/synthetic images
    ├── gen1.jpg
    ├── gen2.png
    └── ...
```

### 3. **Train the Model**

```bash
python src/training/trainer.py
```

The script will:
- Load data from `data/raw/`
- Fine-tune the DINOv2 head for 10 epochs
- Save the best model to `models_saved/dinov2_best.pt`
- Display training progress and validation metrics

### 4. **Evaluate Performance**

```bash
python src/training/evaluate.py
```

Generates a detailed report with:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization
- Per-image latency metrics

### 5. **Run Web Interface**

```bash
python app/main.py
```

Open your browser to `http://localhost:7860` and upload an image for instant AI detection with heatmap visualization.

### 6. **Export to ONNX (Optional)**

For optimized inference and cross-platform deployment:

```bash
python src/models/export_onnx.py
```

This creates `models_saved/visionguard.onnx` which can be deployed with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Load ONNX model
session = ort.InferenceSession("models_saved/visionguard.onnx")

# Prepare input
image = Image.open("test.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).numpy()

# Inference
outputs = session.run(None, {"input": input_tensor})
print("Prediction:", outputs[0])
```

---

## 📋 Project Structure

```
VisionGuardAI/
├── app/
│   └── main.py                 # Gradio web interface
├── src/
│   ├── data/
│   │   └── data_loader.py      # Data pipeline with augmentation
│   ├── models/
│   │   └── model.py            # DINOv2-based classifier
│   ├── inference/
│   │   └── predictor.py        # Inference + GradCAM explainability
│   ├── training/
│   │   ├── trainer.py          # Training loop with early stopping
│   │   └── evaluate.py         # Evaluation metrics & visualization
│   └── utils/
├── configs/
│   └── config.yaml             # Training & model configuration
├── data/
│   ├── raw/                    # Original dataset (REAL/ and FAKE/)
│   ├── processed/              # Preprocessed data
│   └── splits/                 # Train/val splits
├── models_saved/               # Trained model checkpoints
├── notebooks/                  # Jupyter notebooks for exploration
├── docker/                     # Docker configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Configuration

Edit `configs/config.yaml` to customize training:

```yaml
project_name: "VisionGuard_AI"

data:
  raw_path: "data/raw"
  train_dir: "data/raw"
  image_size: 224
  batch_size: 32
  num_workers: 0  # Increase on Linux/Mac for faster data loading

model:
  name: "dinov2_vits14"
  num_classes: 2
  learning_rate: 0.0001
  learning_rate_head: 0.001  # Fine-tuning rate
  epochs: 10
  device: "auto"  # "cuda", "cpu", or "auto"

training:
  save_dir: "models_saved"
  save_best: true
  patience: 3  # Early stopping patience
```

---

## 🧠 Model Architecture

### **Backbone: DINOv2 Vision Transformer**
- Self-supervised pre-trained on 142M images
- Patch size: 14×14, Image size: 224×224
- Output: 384-dimensional feature vectors
- **Status: Frozen** (transfer learning)

### **Classification Head**
```
384-dim features
    ↓
Linear(384 → 256)
    ↓
BatchNorm1d + ReLU
    ↓
Dropout(0.3)
    ↓
Linear(256 → 2)  [REAL/FAKE logits]
```

**Why DINOv2?**
- Superior generalization across diverse image domains
- Learned without human labels (self-supervised)
- Captures semantic features beyond texture patterns
- Ideal for detecting AI artifacts across different generators

---

## 📊 Performance Metrics

### Evaluation Example Output
```
==============================
  📊 VISIONGUARD EVALUATION REPORT
==============================
✅ Accuracy:        94.32%
⚡ Avg Latency:     287.45 ms/image
------------------------------
              precision    recall  f1-score   support
        FAKE     0.9412    0.9231    0.9320       520
        REAL     0.9455    0.9615    0.9534       650
     accuracy                       0.9432      1170
    macro avg     0.9433    0.9423    0.9427      1170
 weighted avg     0.9433    0.9432    0.9432      1170
```

---

## 🎨 Explainability: GradCAM Heatmaps

VisionGuard generates **Gradient-weighted Class Activation Maps** showing exactly which image regions triggered AI detection:

- **Red regions**: Strong AI artifact indicators
- **Blue regions**: Natural/authentic features
- **Green regions**: Neutral regions

This transparency is crucial for understanding model decisions and building user trust.

---

## 🔧 Usage Examples

### **Python API (PyTorch)**

```python
from src.inference.predictor import VisionGuardPredictor

# Initialize predictor
predictor = VisionGuardPredictor("models_saved/dinov2_best.pt")

# Make prediction
result = predictor.predict("test_image.jpg")

print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']}%")
print(f"Probabilities: {result['probabilities']}")
# Returns PIL Image with heatmap
heatmap = result['heatmap']
```

### **ONNX Runtime (Optimized Inference)**

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# Load ONNX model
session = ort.InferenceSession("models_saved/visionguard.onnx")

# Prepare input
image = Image.open("test.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0).numpy()

# Inference
outputs = session.run(None, {"input": input_tensor})
logits = outputs[0]

# Get predictions
probs = F.softmax(torch.from_numpy(logits), dim=1)
confidence, predicted_class = torch.max(probs, 1)

labels = ['FAKE', 'REAL']
print(f"Verdict: {labels[predicted_class.item()]}")
print(f"Confidence: {confidence.item() * 100:.2f}%")
```

### **Command Line**

```bash
# Train
python src/training/trainer.py

# Evaluate
python src/training/evaluate.py

# Export to ONNX
python src/models/export_onnx.py

# Web UI
python app/main.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| torch, torchvision | Deep learning framework |
| numpy, pandas | Data processing |
| scikit-learn | Metrics & evaluation |
| opencv-python | Image manipulation |
| pytorch-grad-cam | Model explainability |
| gradio | Web interface |
| onnxruntime | **ONNX model inference** ⚡ |
| pyyaml | Configuration |
| matplotlib, seaborn | Visualization |

See `requirements.txt` for versions.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Data directory not found"** | Ensure `data/raw/REAL/` and `data/raw/FAKE/` exist |
| **CUDA out of memory** | Reduce `batch_size` in `config.yaml` |
| **Slow data loading** | Increase `num_workers` (Linux/Mac only) |
| **Model won't load** | Check `models_saved/dinov2_best.pt` exists |
| **Gradio port in use** | Change port in `app/main.py` |

---

## 🚢 Deployment

### **ONNX Runtime (Cross-Platform)**

Export to ONNX for optimized inference across platforms:

```bash
python src/models/export_onnx.py
```

Benefits:
- ✅ Hardware acceleration (CPU/GPU/NPU)
- ✅ ~40-50% faster inference
- ✅ Language-agnostic (Python, C++, C#, Java, etc.)
- ✅ Edge device compatible
- ✅ Smaller model size

### **Docker**

```bash
docker build -f docker/Dockerfile -t visionguard:latest .
docker run -p 7860:7860 visionguard:latest
```

### **Cloud (AWS/GCP/Azure)**

1. Build Docker image
2. Push to container registry
3. Deploy on cloud service (Lambda, Cloud Run, Container Instances)

For ONNX deployment on cloud:
```bash
# Export ONNX model
python src/models/export_onnx.py

# Deploy models_saved/visionguard.onnx to your cloud service
```

---

## 📈 Training Tips

- **More Data = Better Model**: Collect diverse AI generators (Stable Diffusion, DALL-E, Midjourney, etc.)
- **Data Augmentation**: Current setup uses ColorJitter & HorizontalFlip. Add more for robustness
- **Class Balance**: Keep REAL/FAKE ratio ~1:1 for best results
- **Early Stopping**: Set `patience: 3` to prevent overfitting
- **Learning Rate**: Use `learning_rate_head: 0.001` for fine-tuning

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/VisionGuardAI/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/yourusername/VisionGuardAI/discussions)
- **Documentation**: See `docs/` folder for detailed guides

---

## 🔗 References

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02055)
- [Gradio Documentation](https://gradio.app/docs/)

---

## 🎓 Citation

If you use VisionGuard AI in your research, please cite:

```bibtex
@software{visionguard_ai_2024,
  title={VisionGuard AI: AI-Generated Image Detection with Explainability},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/VisionGuardAI}
}
```

---

## ⭐ Acknowledgments

- **Meta AI**: For developing DINOv2 vision foundation model
- **Gradio**: For accessible web interface framework
- **PyTorch**: For the excellent deep learning library

---

<div align="center">

**Made with ❤️ for AI transparency and interpretability**

[Star us on GitHub](https://github.com/yourusername/VisionGuardAI) • [Report Issues](https://github.com/yourusername/VisionGuardAI/issues) • [Documentation](docs/)

</div>
