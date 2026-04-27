import gradio as gr
from model import device
from train import load_model
from config import model_path

model = load_model(model_path).eval().to(device)

iface = gr.Interface(
    fn=model.generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Caption Generator",
    description="Upload an image and get a generated caption."
)

if __name__ == "__main__":
    iface.launch(inline=False,server_name="0.0.0.0", server_port=7860,inbrowser=True)