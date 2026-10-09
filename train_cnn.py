"""Train the single-character CNN on EMNIST (or digits-only by choosing a split).
    python train_cnn.py --split balanced --epochs 10
"""
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import CNN, SPLITS, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="balanced", choices=list(SPLITS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    norm = transforms.Normalize((0.1307,), (0.3081,))
    fix = transforms.Lambda(lambda im: im.transpose(Image.TRANSPOSE))   # EMNIST is stored transposed
    train_tf = transforms.Compose([fix, transforms.RandomAffine(10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
                                   transforms.ToTensor(), norm])
    test_tf = transforms.Compose([fix, transforms.ToTensor(), norm])

    train_ds = datasets.EMNIST("data", split=args.split, train=True, download=True, transform=train_tf)
    test_ds = datasets.EMNIST("data", split=args.split, train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers)
    test_dl = DataLoader(test_ds, 512, num_workers=args.workers)

    model = CNN(len(SPLITS[args.split])).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    best = 0.0

    for ep in range(1, args.epochs + 1):
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
        acc = correct / len(test_ds)
        print(f"epoch {ep}: test accuracy {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"state": model.state_dict(), "split": args.split}, f"cnn_{args.split}.pt")
    print(f"best {best:.4f} -> cnn_{args.split}.pt")


if __name__ == "__main__":
    main()
