"""Read handwriting from an image.
    python predict_text.py page.png                       # CRNN (words / sentences)
    python predict_text.py letter.png --mode char         # single character with the CNN
    python predict_text.py word.png --mode cnn            # segment letters, classify each with the CNN
    add --single-line to skip line splitting
"""
import argparse
import torch
import torch.nn.functional as F
from models import CNN, CRNN, SPLITS, DEVICE, greedy_decode
from preprocess import (load_gray, to_ink, crop_to_ink, prep_char, word_to_tensor,
                        split_lines, split_words, split_chars)


def load_cnn(path="cnn_balanced.pt"):
    ck = torch.load(path, map_location=DEVICE)
    labels = SPLITS[ck["split"]]
    m = CNN(len(labels)).to(DEVICE)
    m.load_state_dict(ck["state"])
    return m.eval(), labels


def load_crnn(path="crnn.pt"):
    ck = torch.load(path, map_location=DEVICE)
    m = CRNN(len(ck["charset"])).to(DEVICE)
    m.load_state_dict(ck["state"])
    return m.eval(), ck["charset"]


@torch.no_grad()
def classify_char(ink, cnn, labels, k=3):
    probs = F.softmax(cnn(prep_char(ink).to(DEVICE)), 1)[0].cpu()
    top = probs.topk(k)
    return [(labels[i], p.item()) for p, i in zip(top.values, top.indices)]


@torch.no_grad()
def _word_crnn(word, crnn, charset):
    return greedy_decode(crnn(word_to_tensor(word).to(DEVICE)), charset)[0]


@torch.no_grad()
def _word_cnn(word, cnn, labels):
    crops = split_chars(word)
    if not crops:
        return ""
    idx = cnn(torch.cat([prep_char(c) for c in crops]).to(DEVICE)).argmax(1).tolist()
    return "".join(labels[i] for i in idx)


def read_text(ink, mode="crnn", crnn=None, cnn=None, single_line=False):
    """ink: binary white-on-black page image -> recognised text (lines joined by \\n)."""
    if not ink.any():
        return ""
    lines = [crop_to_ink(ink)] if single_line else split_lines(ink)
    out = []
    for line in lines:
        words = split_words(line)
        if mode == "crnn":
            out.append(" ".join(_word_crnn(w, *crnn) for w in words))
        else:
            out.append(" ".join(_word_cnn(w, *cnn) for w in words))
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mode", default="crnn", choices=["crnn", "cnn", "char"])
    ap.add_argument("--single-line", action="store_true")
    ap.add_argument("--cnn", default="cnn_balanced.pt")
    ap.add_argument("--crnn", default="crnn.pt")
    a = ap.parse_args()

    ink = to_ink(load_gray(a.image))
    if a.mode == "char":
        cnn = load_cnn(a.cnn)
        for lab, p in classify_char(ink, *cnn):
            print(f"{lab}: {p:.2%}")
    elif a.mode == "cnn":
        print(read_text(ink, "cnn", cnn=load_cnn(a.cnn), single_line=a.single_line))
    else:
        print(read_text(ink, "crnn", crnn=load_crnn(a.crnn), single_line=a.single_line))
