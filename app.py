import torch
import gradio as gr
from train import load_model
from config import model_path
from model import model


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(model_path).to(device)
    
iface = gr.Interface(fn=model.generate_caption, 
                     live=True,
                    inputs=gr.Image(type="pil"), 
                    outputs="text",
                    title="Image Caption Generator",
                    description="Upload an image and get a generated caption.")


if __name__ == "__main__":
    iface.launch(inbrowser=True)