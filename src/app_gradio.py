import torch
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, BlipForConditionalGeneration, LlavaForConditionalGeneration


BLIP_ID = "outputs/blip-caption/final" # or 'Salesforce/blip-image-captioning-base'
LLAVA_ID = "llava-hf/llava-1.5-7b-hf"


class Captioner:
    def __init__(self, model_id=BLIP_ID):
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
    def __call__(self, image: Image.Image, max_new_tokens=30):
        enc = self.proc(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=int(max_new_tokens))
        return self.proc.batch_decode(out, skip_special_tokens=True)[0]
    
class VQA:
    def __init__(self, model_id=LLAVA_ID):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            low_cpu_mem_usage=True, load_in_4bit=(device=="cuda"), device_map="auto")
    def __call__(self, image: Image.Image, question: str, max_new_tokens=24):
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        enc = self.proc(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=int(max_new_tokens))
        text = self.proc.decode(out[0], skip_special_tokens=True)
        return text.split("ASSISTANT:")[-1].strip()
    
cap = Captioner()
vqa = VQA()

with gr.Blocks(title="Captioning + VQA (BLIP + LLaVA)") as demo:
    gr.Markdown("# 📷🗣️ Image Captioning & Visual QA\nUpload an image. Choose *Caption* or ask a *Question*.")
    with gr.Row():
        image = gr.Image(label="Image", type="pil")
        with gr.Column():
            tab = gr.Radio(["Caption", "VQA"], value="Caption", label="Mode")
            cap_btn = gr.Button("Generate caption")
            q = gr.Textbox(label="Question (for VQA)")
            vqa_btn = gr.Button("Answer")
            max_new = gr.Slider(8, 64, value=30, step=1, label="Max new tokens")
            out = gr.Textbox(label="Output")

    def on_caption(img, _, max_new_tokens):
        if img is None:
            return "Please upload an image."
        return cap(img, max_new_tokens)
    
    def on_vqa(img, question, max_new_tokens):
        if img is None:
            return "Please upload an image."
        if not question:
            return "Please enter a question."
        return vqa(img, question, max_new_tokens)
    
    cap_btn.click(on_caption, [image, tab, max_new], out)
    vqa_btn.click(on_vqa, [image, q, max_new], out)

if __name__ == "__main__":
    demo.launch()