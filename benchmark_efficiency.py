"""Efficiency benchmark for MFSA-Net and the baselines. No datasets and no trained checkpoints are needed:
latency, throughput, memory and FLOPs depend on the architecture, not on the learned weights.

Run from the repository root on the machine you want to report (e.g. the RTX 4090 pod):
    python3 benchmark_efficiency.py --device cuda > benchmark_gpu.txt
    python3 benchmark_efficiency.py --device cpu --threads 8 --iters 20 > benchmark_cpu.txt
Add --no-pretrained if the machine has no internet access (weights are irrelevant for timing).
Reports: parameters, forward FLOPs/MACs (torch FlopCounterMode, batch 1), inference latency (batch 1) and throughput (batch 32),
peak inference memory, peak training memory and time of one training step at the training batch size (16), and an estimate of the
training time of one run from the step time (data loading excluded)."""
import argparse, json, platform, statistics, sys, time

sys.path.insert(0, ".")
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--size", type=int, default=288)
ap.add_argument("--train-batch", type=int, default=16)
ap.add_argument("--epochs", type=int, default=35, help="typical epochs until early stopping (best epoch 19-33, patience 15)")
ap.add_argument("--no-pretrained", action="store_true")
args = ap.parse_args()

if args.no_pretrained:
    import timm
    _orig = timm.create_model
    timm.create_model = lambda *a, **k: _orig(*a, **{**k, "pretrained": False})
from models_mfsanet import MFSANet
from models_specxnet import SpecXNet
from models_defax import DeFaX
from models_xception import XceptionBaseline

dev = torch.device(args.device)
if dev.type == "cpu":
    torch.set_num_threads(args.threads)
torch.backends.cudnn.deterministic = True   # same settings as train_common.set_seed
torch.backends.cudnn.benchmark = False
S = args.size
registry = {"SpecXNet": lambda: SpecXNet(input_size=S), "Xception": lambda: XceptionBaseline(input_size=S),
            "DeFaX": lambda: DeFaX(input_size=S), "MFSA-Net": lambda: MFSANet(input_size=S, variant="full")}


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), sorted(ts)[int(0.95 * (len(ts) - 1))]


env = {"torch": torch.__version__, "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
       "device": args.device, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "cuda": torch.version.cuda,
       "threads": args.threads if dev.type == "cpu" else None, "input": f"{S}x{S}", "precision": "fp32"}
print("ENVIRONMENT:", json.dumps(env))
rows = []
for name, build in registry.items():
    torch.manual_seed(0)
    model = build().to(dev).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    x1 = torch.randn(1, 3, S, S, device=dev)
    flops = macs = float("nan")
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x1)
        flops = fc.get_total_flops() / 1e9; macs = flops / 2
    except Exception as e:  # pragma: no cover
        print("flop counter failed:", e)
    row = {"model": name, "params_M": round(params, 2), "fwd_GFLOPs(2xMAC)": round(flops, 2), "fwd_GMACs": round(macs, 2)}
    with torch.no_grad():
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        med, p95 = timed(lambda: model(x1), args.warmup, args.iters)
        row["latency_b1_ms_median"], row["latency_b1_ms_p95"] = round(med, 2), round(p95, 2)
        if dev.type == "cuda":
            row["inference_peak_mem_b1_MB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        if dev.type == "cuda":
            xb = torch.randn(32, 3, S, S, device=dev); torch.cuda.reset_peak_memory_stats()
            med32, _ = timed(lambda: model(xb), max(2, args.warmup // 2), max(5, args.iters // 5))
            row["throughput_b32_img_per_s"] = round(32 / (med32 / 1000), 1)
            row["inference_peak_mem_b32_MB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    if dev.type == "cuda":
        model.train(); opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        xt = torch.randn(args.train_batch, 3, S, S, device=dev); yt = torch.randint(0, 2, (args.train_batch,), device=dev)
        def step():
            opt.zero_grad(); loss = nn.CrossEntropyLoss()(model(xt), yt); loss.backward(); opt.step()
        torch.cuda.reset_peak_memory_stats()
        try:
            tmed, _ = timed(step, 3, max(5, args.iters // 5))
            row["train_step_ms_b%d" % args.train_batch] = round(tmed, 1)
            row["train_peak_mem_b%d_GB" % args.train_batch] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            for label, frames in (("zero-shot (86,400 frames/epoch)", 86400), ("domain-adapted (134,240 frames/epoch)", 134240)):
                hours = tmed / 1000 * (frames / args.train_batch) * args.epochs / 3600
                row["est_run_hours_" + label.split(" ")[0]] = round(hours, 1)
        except RuntimeError as e:
            row["train_step"] = "failed: " + str(e)[:80]
        torch.cuda.empty_cache()
    rows.append(row)
    print(json.dumps(row), flush=True)
    del model
print("\nSUMMARY TABLE")
keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
print("\t".join(keys))
for r in rows:
    print("\t".join(str(r.get(k, "")) for k in keys))
print("\nNote: the training-time estimate multiplies the measured step time by the number of steps per epoch and by", args.epochs,
      "epochs; it excludes data loading, validation and checkpointing, so real runs take longer.")
