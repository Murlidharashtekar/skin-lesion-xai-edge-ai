"""XAI comparison (Kaggle run): Grad-CAM (conv_head, bn2) vs Grad-CAM++ (bn2) on the E1 seed-42 checkpoint.
Paths below are the Kaggle paths used for the reported results."""
import json, zlib, base64, io
import numpy as np, pandas as pd, torch, timm
from pathlib import Path
from PIL import Image
from scipy.stats import spearmanr, wilcoxon
from torchvision import transforms
from matplotlib import cm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CK = "/kaggle/working/outputs/E1_effnet_b3/E1_effnet_b3_best.pth"
TEST = "/kaggle/working/repo/data/splits_v2/test.csv"
IMG = "/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000"
CL = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]; C2I = {c: i for i, c in enumerate(CL)}
N_PER = 10
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trap = getattr(np, "trapezoid", None) or np.trapz
rng = np.random.default_rng(0)
g = torch.Generator(device=dev).manual_seed(0)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
norm = lambda x: (x - MEAN) / STD
tf = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])

model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=7)
model.load_state_dict(torch.load(CK, map_location="cpu")); model.to(dev).eval()

paths = {p.stem: p for p in Path(IMG).rglob("*.jpg")}
te = pd.read_csv(TEST)
sub = pd.concat([te[te.dx == c].sample(min(N_PER, (te.dx == c).sum()), random_state=0) for c in CL])
print("images:", len(sub), sub.dx.value_counts().to_dict())

cams = {"gradcam_conv_head": GradCAM(model=model, target_layers=[model.conv_head]),
        "gradcam_bn2": GradCAM(model=model, target_layers=[model.bn2]),
        "gradcampp_bn2": GradCAMPlusPlus(model=model, target_layers=[model.bn2])}

def predict(x):
    with torch.no_grad():
        return torch.softmax(model(norm(x[None])), 1)[0]

def faith(xn, m, c0):
    flat = m.ravel() + 1e-6 * rng.random(m.size)
    order = np.argsort(-flat); rank = np.empty(m.size, dtype=np.int64); rank[order] = np.arange(m.size)
    rank = torch.tensor(rank.reshape(m.shape), device=dev)
    fr = np.linspace(0, 1, 11)
    masks = torch.stack([(rank < int(f * m.size)).float() for f in fr])[:, None]
    with torch.no_grad():
        pdl = torch.softmax(model(xn * (1 - masks)), 1)[:, c0].cpu().numpy()
        pin = torch.softmax(model(xn * masks), 1)[:, c0].cpu().numpy()
    return float(trap(pdl, fr)), float(trap(pin, fr))

rows, maps_keep = [], {}
for _, r in sub.iterrows():
    x = tf(Image.open(paths[r.image_id]).convert("RGB")).to(dev)
    pr = predict(x); c0 = int(pr.argmax())
    xs = {"clean": x,
          "noise": (x + 0.02 * torch.randn(x.shape, generator=g, device=dev)).clamp(0, 1),
          "bright": (x * 1.1).clamp(0, 1)}
    flips = {k: int(predict(v).argmax()) != c0 for k, v in xs.items()}
    xn = norm(x[None])
    for name, cam in cams.items():
        mp = {k: cam(input_tensor=norm(v[None]), targets=[ClassifierOutputTarget(c0)])[0] for k, v in xs.items()}
        st = {}
        for k in ("noise", "bright"):
            a, b = mp["clean"][::6, ::6].ravel(), mp[k][::6, ::6].ravel()
            st[k] = spearmanr(a, b).correlation if a.std() > 0 and b.std() > 0 else np.nan
        d, i = faith(xn, mp["clean"], c0)
        rows.append(dict(image_id=r.image_id, true=r.dx, pred=CL[c0], method=name, stab_noise=st["noise"],
                         stab_bright=st["bright"], del_auc=d, ins_auc=i, flip_noise=flips["noise"], flip_bright=flips["bright"]))
        maps_keep[(r.image_id, name)] = mp["clean"]
    d, i = faith(xn, rng.random((300, 300)), c0)
    rows.append(dict(image_id=r.image_id, true=r.dx, pred=CL[c0], method="random_map", stab_noise=np.nan,
                     stab_bright=np.nan, del_auc=d, ins_auc=i, flip_noise=flips["noise"], flip_bright=flips["bright"]))

df = pd.DataFrame(rows); df.to_csv("/kaggle/working/xai_rows.csv", index=False)
pd.set_option("display.width", 200)
agg = df.groupby("method").agg(stab_noise=("stab_noise", "mean"), stab_bright=("stab_bright", "mean"),
                               del_auc=("del_auc", "mean"), ins_auc=("ins_auc", "mean"))
agg["del_sd"] = df.groupby("method").del_auc.std(); agg["ins_sd"] = df.groupby("method").ins_auc.std()
print("\nSUMMARY (70 images; deletion lower=better, insertion higher=better)\n", agg.round(4))
print("\nprediction flips: noise", int(df[df.method == "random_map"].flip_noise.sum()),
      "| brightness", int(df[df.method == "random_map"].flip_bright.sum()), "of", len(sub))
for col in ("del_auc", "ins_auc", "stab_noise", "stab_bright"):
    a = df[df.method == "gradcam_bn2"].set_index("image_id")[col]
    b = df[df.method == "gradcampp_bn2"].set_index("image_id")[col]
    ok = a.notna() & b.notna()
    print(f"gradcam_bn2 vs gradcampp_bn2 {col}: mean diff {np.mean(a[ok]-b[ok]):+.4f}, Wilcoxon p={wilcoxon(a[ok], b[ok]).pvalue:.3f}")

tiles = []
for c in ("mel", "nv", "bkl", "akiec"):
    r = sub[sub.dx == c].iloc[0]
    base = np.array(Image.open(paths[r.image_id]).convert("RGB").resize((150, 150)))
    row = [base]
    for name in cams:
        m = np.array(Image.fromarray((maps_keep[(r.image_id, name)] * 255).astype(np.uint8)).resize((150, 150))) / 255
        row.append((0.6 * base + 0.4 * 255 * cm.jet(m)[:, :, :3]).astype(np.uint8))
    tiles.append(np.concatenate(row, 1))
buf = io.BytesIO(); Image.fromarray(np.concatenate(tiles, 0)).save(buf, "JPEG", quality=70)
blob = base64.b64encode(buf.getvalue()).decode()
print("\n#TAG xai_fig #LEN", len(blob), "#CRC", zlib.crc32(blob.encode()))
print("# rows: mel, nv, bkl, akiec | cols: original, gradcam_conv_head, gradcam_bn2, gradcampp_bn2")
for i in range(0, len(blob), 24000):
    print(f"#CHUNK {i // 24000 + 1}"); print(blob[i:i + 24000])
print("#END")
