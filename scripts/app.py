import torch
from cellpilot.inference.inference import Inference
from cellpilot.inference.app_tools import App
import gradio as gr
from gradio_image_prompter import ImagePrompter
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir", type=str, default="/vol/data/models/")
parser.add_argument("--model_name", type=str, default="model-ap0xl4l1:v19")
parser.add_argument("--cellvit_model", type=str, default="CellViT-256-x40.pth")


args = parser.parse_args()

inference_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_dir": args.model_dir,
    "model_name": args.model_name,
}
app_config = {
    
}

config = {
    "inference_config": inference_config,
    "app_config": app_config,
}
app = App(config)

with gr.Blocks(theme=gr.themes.Default(text_size="lg")) as demo: 
    with gr.Row():
        whole_image = ImagePrompter(label="Whole Image")
        whole_image.upload(app.load_image, inputs=[whole_image], outputs=[whole_image])
        with gr.Column(visible=True) as initial_buttons:
            auto_segment_btn = gr.Button("Auto Segment")
            auto_segment_btn.click(app.segment_automatically_app, inputs=[], outputs=[whole_image])
            add_mask_btn = gr.Button("Add Mask",)
            add_mask_btn.click(app.add_mask, inputs=[whole_image], outputs=[whole_image])
            start_refine_mask_btn = gr.Button("Refine Mask")
            remove_mask_btn = gr.Button("Remove Mask")
            remove_mask_btn.click(app.remove_mask, inputs=[whole_image], outputs=[whole_image])
        with gr.Column(visible=False) as refine_buttons:
            refine_mask_btn = gr.Button("Refine")
            refine_mask_btn.click(app.refine_mask, inputs=[whole_image], outputs=[whole_image])
            finish_mask_btn = gr.Button("Finish Mask")  
        start_refine_mask_btn.click(app.start_refine_mask, inputs=[whole_image], outputs=[whole_image, refine_buttons, initial_buttons])
        finish_mask_btn.click(app.finish_mask, inputs=[], outputs=[whole_image, initial_buttons, refine_buttons])
    with gr.Row():
        amount = gr.Number(value=100, label="Step Size", visible = False)
        #amount = 100
        with gr.Column():
            left_button = gr.Button(value="\U0001F814")
            left_button.click(app.move_left, inputs=[amount], outputs=[whole_image])
        with gr.Column():
            up_button = gr.Button(value="\U0001F815")
            up_button.click(app.move_up, inputs=[amount], outputs=[whole_image])
            down_button = gr.Button(value="\U0001F817")
            down_button.click(app.move_down, inputs=[amount], outputs=[whole_image])
        with gr.Column():
            right_button = gr.Button(value="\U0001F816")
            right_button.click(app.move_right, inputs=[amount], outputs=[whole_image])
        zoom_bar = gr.Slider(minimum=1, maximum=5, step=1, label="Zoom Factor", value=1)
        zoom_bar.release(app.zoom, inputs=[zoom_bar, whole_image], outputs=[whole_image])
demo.launch(share=True)

