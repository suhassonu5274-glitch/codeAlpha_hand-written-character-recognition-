"""Models + charsets shared by every script."""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Label order of the EMNIST splits
BYCLASS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"   # 62 classes
BALANCED = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"                  # 47 classes
SPLITS = {"byclass": BYCLASS, "balanced": BALANCED}


class CNN(nn.Module):
    """Single-character classifier for 28x28 grayscale images."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32)
        self.conv2, self.bn2 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64)
        self.conv3, self.bn3 = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)   # 14x14
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)   # 7x7
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)   # 3x3
        x = self.drop(torch.flatten(x, 1))
        return self.fc2(self.drop(F.relu(self.fc1(x))))


class CRNN(nn.Module):
    """CNN feature extractor -> BiLSTM -> per-timestep class scores (CTC).
    Input : (B, 1, 32, W)  with W a multiple of 4
    Output: (B, W/4, num_classes + 1)   log-probs, index 0 = CTC blank
    """

    def __init__(self, num_classes, hidden=256):
        super().__init__()

        def blk(i, o):
            return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]

        self.cnn = nn.Sequential(
            *blk(1, 64), nn.MaxPool2d(2, 2),                       # 16 x W/2
            *blk(64, 128), nn.MaxPool2d(2, 2),                     # 8  x W/4
            *blk(128, 256), *blk(256, 256), nn.MaxPool2d((2, 1)),  # 4  x W/4
            *blk(256, 512), nn.MaxPool2d((2, 1)),                  # 2  x W/4
            nn.Conv2d(512, 512, (2, 1)), nn.ReLU(True),            # 1  x W/4
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, num_classes + 1)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)   # (B, T, 512)
        out, _ = self.rnn(f)
        return self.fc(out).log_softmax(2)


def greedy_decode(log_probs, charset):
    """CTC greedy decoding: collapse repeats, drop blanks (index 0)."""
    results = []
    for seq in log_probs.argmax(2).cpu().numpy():
        out, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                out.append(charset[k - 1])
            prev = k
        results.append("".join(out))
    return results
