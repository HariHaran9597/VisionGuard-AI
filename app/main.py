import gradio as gr
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.predictor import VisionGuardPredictor

# 1. Load Model
model_path = "models_saved/dinov2_best.pt"
print(f"⏳ Loading VisionGuard AI ({model_path})...")

try:
    predictor = VisionGuardPredictor(model_path)
    print("✅ System Ready.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# 2. Logic
def analyze_image(image):
    if image is None:
        return None, None, "Please upload an image."
    
    temp_path = "temp_analysis.jpg"
    image.save(temp_path)
    
    try:
        # Run Prediction
        result = predictor.predict(temp_path)
        
        summary = (
            f"Verdict: {result['verdict']}\n"
            f"Confidence: {result['confidence']}%"
        )
        
        # Return: Label Dict, Heatmap Image, Summary Text
        return result['probabilities'], result['heatmap'], summary
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# 3. UI
with gr.Blocks(title="VisionGuard AI") as demo:
    gr.Markdown("# 🛡️ VisionGuard AI")
    gr.Markdown("Upload an image to detect AI artifacts. The **Heatmap** shows which areas triggered the detection.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Source")
            submit_btn = gr.Button("Analyze Integrity", variant="primary")
            
        with gr.Column():
            # Output 1: Probability
            label_output = gr.Label(num_top_classes=2, label="Probability")
            # Output 2: Heatmap
            heatmap_output = gr.Image(label="Attention Heatmap (X-Ray)")
            # Output 3: Text
            info_output = gr.Textbox(label="Verdict")
            
    submit_btn.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[label_output, heatmap_output, info_output]
    )

if __name__ == "__main__":
    demo.launch()