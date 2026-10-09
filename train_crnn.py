"""Train a CRNN + CTC word recogniser on SYNTHETIC words built by stitching EMNIST characters.
No extra dataset needed.   python train_crnn.py --epochs 15 --samples 20000

For real handwriting (IAM etc.) replace SyntheticWords with a Dataset that returns
(uint8 image of height 32, text) and keep the rest unchanged.
"""
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from models import CRNN, BYCLASS, DEVICE, greedy_decode

H = 32


class SyntheticWords(Dataset):
    def __init__(self, emnist, charset, n_samples, min_len=1, max_len=8):
        self.data = emnist.data.numpy()
        targets = emnist.targets.numpy()
        self.charset, self.n, self.min_len, self.max_len = charset, n_samples, min_len, max_len
        self.by_class = [np.where(targets == i)[0] for i in range(len(charset))]

    def __len__(self):
        return self.n

    def render(self, text, rng):
        pieces = [np.zeros((H, 6), np.uint8)]
        for ch in text:
            idx = rng.choice(self.by_class[self.charset.index(ch)])
            img = self.data[idx].T                                    # upright 28x28
            cols = np.where(img.max(0) > 40)[0]
            if len(cols):
                img = img[:, cols.min():cols.max() + 1]               # trim horizontally only (keeps case info)
            w = max(4, int(img.shape[1] * H / 28 * rng.uniform(0.9, 1.1)))
            pieces.append(cv2.resize(img, (w, H), interpolation=cv2.INTER_AREA))
            pieces.append(np.zeros((H, int(rng.integers(1, 8))), np.uint8))
        canvas = np.hstack(pieces)
        s = rng.uniform(-0.3, 0.3)                                     # random slant
        w2 = canvas.shape[1] + int(abs(s) * H) + 1
        M = np.float32([[1, s, -s * H / 2 + abs(s) * H / 2], [0, 1, 0]])
        canvas = cv2.warpAffine(canvas, M, (w2, H))
        if rng.random() < 0.3:                                         # thicker strokes sometimes
            canvas = cv2.dilate(canvas, np.ones((2, 2), np.uint8))
        return canvas

    def __getitem__(self, i):
        rng = np.random.default_rng()
        L = int(rng.integers(self.min_len, self.max_len + 1))
        text = "".join(self.charset[j] for j in rng.integers(0, len(self.charset), L))
        return self.render(text, rng), text


def make_collate(charset):
    def collate(batch):
        imgs, texts = zip(*batch)
        W = (max(i.shape[1] for i in imgs) + 3) // 4 * 4
        x = torch.full((len(imgs), 1, H, W), -1.0)
        for k, im in enumerate(imgs):
            x[k, 0, :, :im.shape[1]] = (torch.from_numpy(im).float() / 255 - 0.5) / 0.5
        targets = torch.tensor([charset.index(c) + 1 for t in texts for c in t])   # +1: 0 is CTC blank
        tgt_lens = torch.tensor([len(t) for t in texts])
        in_lens = torch.full((len(imgs),), W // 4, dtype=torch.long)
        return x, targets, in_lens, tgt_lens, texts
    return collate


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--samples", type=int, default=20000, help="synthetic words per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    charset = BYCLASS
    tr = datasets.EMNIST("data", split="byclass", train=True, download=True)
    te = datasets.EMNIST("data", split="byclass", train=False, download=True)
    collate = make_collate(charset)
    train_dl = DataLoader(SyntheticWords(tr, charset, args.samples), args.batch, num_workers=args.workers, collate_fn=collate)
    val_dl = DataLoader(SyntheticWords(te, charset, 1000), 100, num_workers=args.workers, collate_fn=collate)

    model = CRNN(len(charset)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    best_cer = 9.9

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, tg, il, tl, _ in train_dl:
            x = x.to(DEVICE)
            out = model(x)                                                     # (B,T,K)
            loss = F.ctc_loss(out.permute(1, 0, 2), tg, il, tl, blank=0, zero_infinity=True)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        errs = chars = exact = n = 0
        with torch.no_grad():
            for x, _, _, _, texts in val_dl:
                preds = greedy_decode(model(x.to(DEVICE)), charset)
                for p, t in zip(preds, texts):
                    errs += edit_distance(p, t); chars += len(t); exact += (p == t); n += 1
        cer = errs / chars
        print(f"epoch {ep}: loss {loss.item():.3f}  CER {cer:.3f}  word-acc {exact / n:.3f}")
        if cer < best_cer:
            best_cer = cer
            torch.save({"state": model.state_dict(), "charset": charset}, "crnn.pt")
    print(f"best CER {best_cer:.3f} -> crnn.pt")


if __name__ == "__main__":
    main()
