"""
Handwritten Character Recognition with a CNN (EMNIST / MNIST)

Install:  pip install torch torchvision pillow numpy
Train:    python handwritten_cnn.py train
Predict:  python handwritten_cnn.py predict my_char.png

Splits: "balanced" (47 classes: digits + merged letters), "byclass" (62 classes),
        "letters" (26 classes), "digits" (10 classes). To use plain MNIST, swap
        the dataset class below for datasets.MNIST and set NUM_CLASSES = 10.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

SPLIT = "balanced"
NUM_CLASSES = 47
# Label order for the EMNIST "balanced" split
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "emnist_cnn.pt"


class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)  # 14x14
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)  # 7x7
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)  # 3x3
        x = self.drop(torch.flatten(x, 1))
        return self.fc2(self.drop(F.relu(self.fc1(x))))


# EMNIST images are stored transposed relative to MNIST; fix orientation.
fix_orientation = transforms.Lambda(lambda img: img.transpose(Image.TRANSPOSE))
norm = transforms.Normalize((0.1307,), (0.3081,))


def train(epochs=8):
    train_tf = transforms.Compose([
        fix_orientation,
        transforms.RandomAffine(10, translate=(0.08, 0.08), scale=(0.9, 1.1)),  # augmentation
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([fix_orientation, transforms.ToTensor(), norm])

    train_ds = datasets.EMNIST("data", split=SPLIT, train=True, download=True, transform=train_tf)
    test_ds = datasets.EMNIST("data", split=SPLIT, train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=512)

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        sched.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x.to(DEVICE)).argmax(1).cpu() == y).sum().item()
        print(f"epoch {epoch}: test accuracy {correct / len(test_ds):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("saved", MODEL_PATH)


def preprocess(path):
    """Turn a photo/scan of ONE character into a 28x28 EMNIST-style tensor."""
    img = ImageOps.grayscale(Image.open(path))
    arr = np.array(img)
    if arr.mean() > 127:                      # dark ink on white paper -> invert
        arr = 255 - arr
    ys, xs = np.where(arr > 60)               # crop tightly around the stroke
    if len(xs) == 0:
        raise ValueError("No character found in image")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    img = Image.fromarray(arr)
    img.thumbnail((20, 20))                   # fit in 20x20 like MNIST/EMNIST
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))  # center
    x = transforms.ToTensor()(canvas)
    return norm(x).unsqueeze(0)               # already upright: no transpose here


def predict(path):
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(preprocess(path).to(DEVICE)), dim=1)[0].cpu()
    top = probs.topk(3)
    for p, i in zip(top.values, top.indices):
        print(f"{LABELS[i]}: {p:.2%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 2 and sys.argv[1] == "predict":
        predict(sys.argv[2])
    else:
        print(__doc__)
