train.py
import argparse, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
C2I = {c: i for i, c in enumerate(CLASSES)}
SIZE = 300
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def tf_train():
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def tf_eval():
    return transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class HAM(Dataset):
    def __init__(self, df, paths, tf):
        self.df = df.reset_index(drop=True)
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.paths[r["image_id"]]).convert("RGB")
        return self.tf(img), C2I[r["dx"]]


# ---------------- models (all take the same 300x300 input) ----------------
class ViTS(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.net = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=7)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.net(x)


class Fusion(nn.Module):
    """EfficientNet-B3 (local) + DeiT-S (global), concatenated pooled features."""
    def __init__(self, pretrained):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=0)
        self.vit = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=0)
        d = self.cnn.num_features + self.vit.num_features
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 7))

    def forward(self, x):
        f1 = self.cnn(x)
        f2 = self.vit(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False))
        return self.head(torch.cat([f1, f2], dim=1))


def build(name, pretrained=True):
    if name == "efficientnet_b3":
        return timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=7)
    if name == "deit_small":
        return ViTS(pretrained)
    if name == "fusion":
        return Fusion(pretrained)
    raise ValueError(name)


def metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_precision": float(precision_score(y, p, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, p, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
    }


@torch.no_grad()
def predict(model, loader, dev, criterion=None):
    model.eval()
    probs, ys, tot = [], [], 0.0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            out = model(x)
        out = out.float()
        if criterion is not None:
            tot += criterion(out, y).item() * x.size(0)
        probs.append(torch.softmax(out, 1).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys), tot / len(loader.dataset)


def dump_preds(path, df, probs, ys):
    out = df[["image_id", "lesion_id", "dx"]].reset_index(drop=True).copy()
    out["y_true"], out["y_pred"] = ys, probs.argmax(1)
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, i]
    out.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["efficientnet_b3", "deit_small", "fusion"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", default="/kaggle/working/outputs")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use N rows per split")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out_dir) / a.tag
    out.mkdir(parents=True, exist_ok=True)

    paths = {p.stem: p for p in Path(a.image_root).rglob("*.jpg")}
    print("images found:", len(paths), "| device:", dev)
    sd = Path(a.split_dir)
    tr, va, te = (pd.read_csv(sd / f"{n}.csv") for n in ("train", "validation", "test"))
    if a.limit:
        tr, va, te = tr.sample(a.limit, random_state=0), va.sample(a.limit, random_state=0), te.sample(a.limit, random_state=0)
    for d in (tr, va, te):
        missing = [i for i in d["image_id"] if i not in paths]
        assert not missing, f"missing images: {missing[:3]}"
    print("train/val/test:", len(tr), len(va), len(te))

    mk = lambda d, t, sh: DataLoader(HAM(d, paths, t), batch_size=a.batch_size, shuffle=sh,
                                     num_workers=a.workers, pin_memory=dev.type == "cuda")
    tl, vl, tel = mk(tr, tf_train(), True), mk(va, tf_eval(), False), mk(te, tf_eval(), False)

    model = build(a.model).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    w = None
    if a.class_weights:
        cnt = tr["dx"].value_counts().reindex(CLASSES).values.astype(float)
        w = torch.tensor(cnt.sum() / (len(CLASSES) * cnt), dtype=torch.float32).to(dev)
        print("class weights:", dict(zip(CLASSES, w.cpu().numpy().round(3))))
    crit = nn.CrossEntropyLoss(weight=w)
    crit_eval = nn.CrossEntropyLoss()  # unweighted, so val loss is comparable across runs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    best_f1, best_ep, bad, hist = -1.0, 0, 0, []
    ckpt = out / f"{a.tag}_best.pth"
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        te0 = time.time()
        model.train()
        run, ys, ps = 0.0, [], []
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                o = model(x)
            loss = crit(o.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0)
            ys.append(y.cpu().numpy()); ps.append(o.argmax(1).cpu().numpy())
        trm = metrics(np.concatenate(ys), np.concatenate(ps))
        vp, vy, vloss = predict(model, vl, dev, crit_eval)
        vm = metrics(vy, vp.argmax(1))
        sched.step(vm["macro_f1"])
        rec = {"epoch": ep, "train_loss": run / len(tl.dataset), "train_accuracy": trm["accuracy"],
               "train_macro_f1": trm["macro_f1"], "val_loss": vloss, "val_accuracy": vm["accuracy"],
               "val_macro_f1": vm["macro_f1"], "learning_rate": opt.param_groups[0]["lr"],
               "epoch_seconds": time.time() - te0}
        hist.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
        if vm["macro_f1"] > best_f1:
            best_f1, best_ep, bad = vm["macro_f1"], ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break
    train_secs = time.time() - t0
    (out / "history.json").write_text(json.dumps(hist, indent=2))

    # Final evaluation: best-validation checkpoint; the test set is used exactly once here.
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    vp, vy, _ = predict(model, vl, dev)
    tp, ty, _ = predict(model, tel, dev)
    dump_preds(out / "val_predictions.csv", va, vp, vy)
    dump_preds(out / "test_predictions.csv", te, tp, ty)
    res = {
        "tag": a.tag, "model": a.model, "seed": a.seed, "epochs_run": len(hist), "best_epoch": best_ep,
        "best_val_macro_f1": best_f1, "params": n_params,
        "fp32_weights_mb": round(ckpt.stat().st_size / 1e6, 2), "train_seconds": round(train_secs, 1),
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__, "timm": timm.__version__,
        "hyperparameters": {k: v for k, v in vars(a).items()},
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "test": metrics(ty, tp.argmax(1)),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()