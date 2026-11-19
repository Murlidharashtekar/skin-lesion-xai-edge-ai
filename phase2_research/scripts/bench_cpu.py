import time, io, json, torch, timm, torch.nn as nn, torch.nn.functional as F, psutil, os
torch.set_num_threads(4); torch.manual_seed(0)

class ViTS(nn.Module):
    def __init__(s):
        super().__init__(); s.net = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=7)
    def forward(s, x): return s.net(F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False))
class Fusion(nn.Module):
    def __init__(s):
        super().__init__()
        s.cnn = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
        s.vit = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
        d = s.cnn.num_features + s.vit.num_features
        s.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 7))
    def forward(s, x):
        return s.head(torch.cat([s.cnn(x), s.vit(F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False))], 1))

def size_mb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.getbuffer().nbytes / 1e6
def bench(m, n=30, warm=5):
    m.eval(); x = torch.randn(1, 3, 300, 300); ts = []
    with torch.no_grad():
        for _ in range(warm): m(x)
        for _ in range(n):
            t = time.perf_counter(); m(x); ts.append((time.perf_counter() - t) * 1000)
    ts.sort(); return ts[len(ts)//2], ts[int(len(ts)*0.95)-1]

models = {"efficientnet_b3_fp32": timm.create_model("efficientnet_b3", pretrained=False, num_classes=7),
          "deit_small_fp32": ViTS(), "fusion_fp32": Fusion()}
models["efficientnet_b3_dynq_int8"] = torch.ao.quantization.quantize_dynamic(
    timm.create_model("efficientnet_b3", pretrained=False, num_classes=7).eval(), {nn.Linear}, dtype=torch.qint8)
models["fusion_dynq_int8"] = torch.ao.quantization.quantize_dynamic(Fusion().eval(), {nn.Linear}, dtype=torch.qint8)

res = {}
for k, m in models.items():
    m.eval(); p50, p95 = bench(m)
    res[k] = dict(params_m=round(sum(p.numel() for p in m.parameters()) / 1e6, 2), size_mb=round(size_mb(m), 1),
                  p50_ms=round(p50, 1), p95_ms=round(p95, 1), rss_mb=round(psutil.Process(os.getpid()).memory_info().rss / 1e6))
    print(k, res[k])
json.dump(res, open("phase2_research/outputs/eval/cpu_benchmark.json", "w"), indent=2)
print("torch", torch.__version__, "| threads 4 | batch 1 | input 300x300 | 30 runs after 5 warmup")
