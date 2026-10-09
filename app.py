"""Draw-and-recognise web demo.   python app.py   (needs trained cnn_balanced.pt and crnn.pt)"""
import os
import gradio as gr
from preprocess import load_gray, to_ink
from predict_text import load_cnn, load_crnn, classify_char, read_text

CNN = load_cnn("cnn_balanced.pt") if os.path.exists("cnn_balanced.pt") else None
CRNN_PACK = load_crnn("crnn.pt") if os.path.exists("crnn.pt") else None


def char_fn(img):
    if CNN is None:
        return {"Train first: python train_cnn.py": 1.0}
    ink = to_ink(load_gray(img))
    if not ink.any():
        return {}
    return {lab: p for lab, p in classify_char(ink, *CNN, k=5)}


def word_fn(img):
    if CRNN_PACK is None:
        return "Train first: python train_crnn.py"
    return read_text(to_ink(load_gray(img)), "crnn", crnn=CRNN_PACK, single_line=True)


def pad():
    return gr.Sketchpad(type="numpy", canvas_size=(420, 220), brush=gr.Brush(default_size=10, colors=["#000000"]))


with gr.Blocks(title="Handwritten Character Recognition") as demo:
    gr.Markdown("# Handwritten Character Recognition")
    with gr.Tab("Single character (CNN)"):
        a, b = pad(), gr.Label(num_top_classes=5)
        gr.Button("Recognise").click(char_fn, a, b)
    with gr.Tab("Word / text line (CRNN)"):
        c, d = pad(), gr.Textbox(label="Recognised text")
        gr.Button("Recognise").click(word_fn, c, d)

if __name__ == "__main__":
    demo.launch()
