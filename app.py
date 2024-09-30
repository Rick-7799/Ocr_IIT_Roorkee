import gradio as gr
from transformers import Qwen2VLForImageAndText, Qwen2VLProcessor
from byaldi import RAGMultiModalModel

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
model = Qwen2VLForImageAndText.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def process_input(image_path, text):
    image = processor(images=image_path, return_tensors="pt")
    inputs = processor(text, return_tensors="pt")
    inputs["image"] = image["pixel_values"]
    return inputs

def perform_ocr(image_path, text, keyword):
    inputs = process_input(image_path, text)
    outputs = RAG(**inputs)
    ocr_text = outputs['generated_text']  # Assuming the output contains generated text
    keyword_found = keyword in ocr_text
    return ocr_text, keyword_found

demo = gr.Interface(
    fn=perform_ocr,
    inputs=["image", "text", gr.Textbox(placeholder="Enter keyword to search")],
    outputs=["text", "label"],
    title="ColPali OCR Demo with Keyword Search",
    description="Perform OCR using ColPali and Qwen2-VL with keyword search functionality."
)

demo.launch()
