"""Image-processing utilities: binarise, crop, segment lines / words / characters."""
import cv2
import numpy as np
import torch


def load_gray(src):
    """Path, numpy array, or Gradio sketch dict -> uint8 grayscale image."""
    if isinstance(src, str):
        g = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(src)
        return g
    if isinstance(src, dict):
        src = src.get("composite", src.get("image"))
    arr = np.asarray(src)
    if arr.ndim == 3 and arr.shape[2] == 4:               # RGBA -> composite on white
        a = arr[..., 3:4].astype(np.float32) / 255
        arr = (arr[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)


def to_ink(gray):
    """Binarise with Otsu; result is white ink (255) on black (0) regardless of input polarity."""
    if gray.std() < 5:                                     # blank page
        return np.zeros_like(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:                                    # mostly white => paper is white => invert
        th = 255 - th
    return th


def crop_to_ink(img, margin=0):
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return img
    y0, y1 = max(ys.min() - margin, 0), ys.max() + 1 + margin
    x0, x1 = max(xs.min() - margin, 0), xs.max() + 1 + margin
    return img[y0:y1, x0:x1]


# ---------- single character (for the CNN) ----------
def prep_char(ink):
    """Binary ink image of ONE character -> normalised (1,1,28,28) tensor (EMNIST style)."""
    ink = crop_to_ink(ink)
    h, w = ink.shape
    k = max(1, int(0.08 * max(h, w))) | 1                  # thicken thin strokes to EMNIST-like width
    ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.uint8)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    t = torch.from_numpy(canvas).float() / 255
    return ((t - 0.1307) / 0.3081)[None, None]


# ---------- word image (for the CRNN) ----------
def word_to_tensor(word_ink, H=32):
    """Binary ink image of one word -> normalised (1,1,32,W) tensor, W multiple of 4."""
    h, w = word_ink.shape
    pv, ph = max(2, int(0.12 * h)), max(4, int(0.12 * h))
    img = cv2.copyMakeBorder(word_ink, pv, pv, ph, ph, cv2.BORDER_CONSTANT, value=0)
    new_w = max(16, int(round(img.shape[1] * H / img.shape[0])))
    new_w = (new_w + 3) // 4 * 4
    img = cv2.resize(img, (new_w, H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float() / 255
    return ((t - 0.5) / 0.5)[None, None]


# ---------- segmentation ----------
def _runs(mask, min_len, merge_gap=0):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i]); start = None
    if start is not None:
        runs.append([start, len(mask)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= min_len]


def split_lines(ink):
    """Horizontal projection profile -> list of line images."""
    proj = (ink > 0).sum(1)
    runs = _runs(proj > 0.003 * ink.shape[1], min_len=8, merge_gap=max(4, int(0.03 * ink.shape[0])))
    return [crop_to_ink(ink[a:b]) for a, b in runs] or [crop_to_ink(ink)]


def split_words(line, gap_ratio=0.35):
    """Dilate horizontally so letters of a word merge, then take connected blobs as words."""
    h = line.shape[0]
    k = max(5, int(gap_ratio * h))
    dil = cv2.dilate(line, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted((cv2.boundingRect(c) for c in cnts), key=lambda b: b[0])
    words = [crop_to_ink(line[:, x:x + w]) for x, y, w, hh in boxes if w * hh >= 30]
    return words or [line]


def split_chars(word):
    """Connected components, merged when they overlap horizontally (e.g. dot of 'i')."""
    cnts, _ = cv2.findContours(word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted((cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 10), key=lambda b: b[0])
    merged = []
    for x, y, w, h in boxes:
        if merged and x < merged[-1][0] + merged[-1][1]:
            px, pw = merged[-1]
            merged[-1] = (px, max(px + pw, x + w) - px)
        else:
            merged.append((x, w))
    return [crop_to_ink(word[:, x:x + w]) for x, w in merged]
