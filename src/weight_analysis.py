"""Ağırlık analizi: öğrenme modelin neresinde oldu, rank gerçekten kullanılıyor mu?

Eğitim sonucu her lineer katmanda bir değişim matrisi var: ΔW = W_sonra - W_önce
  - full FT:  ΔW = W_ft - W_base               (doğrudan ağırlık farkı)
  - LoRA:     ΔW = (alpha / r) * B @ A          (adapter'dan hesaplanır, model yüklemeye gerek yok)

Bu ΔW'lerden iki şey ölçüyoruz:
  1. Göreli değişim  ||ΔW|| / ||W||   -> hangi katman/modül ne kadar değişti?
  2. Etkin rank      ΔW'nin enerjisinin (σ² toplamı) %90'ını kaç tekil değer taşıyor?
     LoRA r=64 verdiysek ve etkin rank 5 çıkıyorsa, 64 boyutun çoğu boşa gidiyor demektir.

GPU gerekmez, CPU'da birkaç dakika sürer.

Kullanım:
    python src/weight_analysis.py --runs_dir <outputs/seed42 klasörü>
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B"
N_LAYERS = 24
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN = MODULES[:4]
ENERGY = 0.90

# plots.py ile aynı görsel dil
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def weight_key(layer: int, module: str) -> str:
    block = "self_attn" if module in ATTN else "mlp"
    return f"model.layers.{layer}.{block}.{module}.weight"


def load_tensors(path: Path) -> dict:
    with safe_open(str(path), "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def effective_rank(delta: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Enerjinin %90'ını taşıyan tekil değer sayısı ve normalize spektrum (σ / σ_max)."""
    s = torch.linalg.svdvals(delta.float())
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    k = int((energy < ENERGY).sum().item()) + 1
    return k, s / s[0]


# ---------------------------------------------------------------------------
# ΔW kaynakları
# ---------------------------------------------------------------------------
def lora_deltas(adapter_dir: Path):
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    t = load_tensors(adapter_dir / "adapter_model.safetensors")
    pat = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.lora_A\.weight$")
    for key, A in t.items():
        m = pat.search(key)
        if not m:
            continue
        B = t[key.replace("lora_A", "lora_B")]
        yield int(m[1]), m[2], scale * (B.float() @ A.float())


def full_deltas(model_dir: Path, base: dict):
    ft = load_tensors(model_dir / "model.safetensors")
    for layer in range(N_LAYERS):
        for module in MODULES:
            k = weight_key(layer, module)
            yield layer, module, ft[k].float() - base[k].float()


# ---------------------------------------------------------------------------
# Analiz
# ---------------------------------------------------------------------------
def analyze(deltas, base: dict, with_rank: bool) -> dict:
    rel = torch.zeros(N_LAYERS, len(MODULES))
    ranks, spectra = {}, {}
    for layer, module, d in deltas:
        j = MODULES.index(module)
        rel[layer, j] = d.norm() / base[weight_key(layer, module)].float().norm()
        if with_rank:
            k, spec = effective_rank(d)
            ranks[(layer, module)] = k
            spectra[(layer, module)] = spec[:64]
    return {"rel": rel, "ranks": ranks, "spectra": spectra}


def style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)


def plot_heatmap(rel: torch.Tensor, title: str, out: Path):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(6.5, 7.5), facecolor=SURFACE)
    im = ax.imshow(rel.numpy() * 100, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(MODULES)), MODULES, rotation=45, ha="right", fontsize=9, color=INK2)
    ax.set_yticks(range(0, N_LAYERS, 2), range(0, N_LAYERS, 2), fontsize=9, color=INK2)
    ax.set_ylabel("katman (0 = girişe en yakın)", fontsize=9, color=INK2)
    ax.axvline(3.5, color=SURFACE, linewidth=3)  # attention | MLP ayracı
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("göreli değişim ||ΔW|| / ||W|| (%)", fontsize=9, color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_layer_profile(results: dict, out: Path):
    """Katman başına göreli değişim: full FT ve LoRA aynı grafikte, kendi maksimumuna göre normalize."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax, "Değişim katmanlara nasıl dağılıyor? (her eğri kendi en büyük değerine bölündü)")
    for name, color in [("full_ft", C3), ("lora_all_r64", C2), ("lora_all_r16", C1)]:
        if name not in results:
            continue
        prof = results[name]["rel"].mean(dim=1)
        prof = prof / prof.max()
        ax.plot(range(N_LAYERS), prof.numpy(), color=color, linewidth=2, marker="o", markersize=4,
                label={"full_ft": "full FT", "lora_all_r64": "LoRA tümü r64", "lora_all_r16": "LoRA tümü r16"}[name])
    ax.set_xlabel("katman (0 = girişe en yakın, 23 = çıkışa en yakın)", fontsize=9, color=INK2)
    ax.set_ylabel("göreli değişim (normalize)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_spectra(results: dict, out: Path):
    """Ortalama normalize tekil değer spektrumu: rank'in ne kadarı gerçekten kullanılıyor?"""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style(ax, "ΔW tekil değerleri (tüm katman ve modüllerin ortalaması)")
    for name, color, label in [("lora_all_r4", "#8a8984", "LoRA r4"), ("lora_all_r16", C1, "LoRA r16"),
                               ("lora_all_r64", C2, "LoRA r64"), ("full_ft", C3, "full FT (ilk 64)")]:
        if name not in results or not results[name]["spectra"]:
            continue
        specs = list(results[name]["spectra"].values())
        n = min(len(s) for s in specs)
        mean = torch.stack([s[:n] for s in specs]).mean(0)
        ax.plot(range(1, n + 1), mean.numpy(), color=color, linewidth=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("tekil değer sırası", fontsize=9, color=INK2)
    ax.set_ylabel("σ_i / σ_1 (log)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--out_dir", default="results/weights")
    args = p.parse_args()
    runs, out = Path(args.runs_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("base ağırlıkları yükleniyor...")
    base = load_tensors(Path(hf_hub_download(MODEL_ID, "model.safetensors")))

    results = {}
    for d in sorted(runs.iterdir()):
        model_dir = d / "model"
        if not model_dir.exists():
            continue
        name = d.name
        print(f"== {name}")
        if (model_dir / "adapter_config.json").exists():
            results[name] = analyze(lora_deltas(model_dir), base, with_rank=True)
        else:
            results[name] = analyze(full_deltas(model_dir, base), base, with_rank=True)

    # --- tablo ---
    lines = ["# Ağırlık analizi", "",
             f"Etkin rank: ΔW enerjisinin %{int(ENERGY * 100)}'ını taşıyan tekil değer sayısı (katman/modül ortalaması).", "",
             "| deney | ort. göreli değişim | alt yarı / üst yarı | attention / MLP | etkin rank (ort.) | verilen rank |",
             "|---|---:|---:|---:|---:|---:|"]
    summary = {}
    for name, r in results.items():
        rel = r["rel"]
        touched = rel > 0
        mean_rel = rel[touched].mean().item()
        lo = rel[:12][touched[:12]].mean().item() if touched[:12].any() else float("nan")
        hi = rel[12:][touched[12:]].mean().item() if touched[12:].any() else float("nan")
        at = rel[:, :4][touched[:, :4]].mean().item() if touched[:, :4].any() else float("nan")
        ml = rel[:, 4:][touched[:, 4:]].mean().item() if touched[:, 4:].any() else float("nan")
        eff = sum(r["ranks"].values()) / len(r["ranks"])
        given = re.search(r"r(\d+)$", name)
        summary[name] = dict(mean_rel=mean_rel, lower=lo, upper=hi, attn=at, mlp=ml, eff_rank=eff,
                             rel_by_layer=rel.mean(1).tolist())
        lines.append(f"| {name} | {mean_rel:.2%} | {lo:.2%} / {hi:.2%} | {at:.2%} / {ml:.2%} | "
                     f"{eff:.1f} | {given[1] if given else 'sınırsız'} |")
    (out / "weights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "weights.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))

    # --- grafikler ---
    for name, title in [("full_ft", "full FT: hangi ağırlık ne kadar değişti?"),
                        ("lora_all_r16", "LoRA tümü r16: hangi ağırlık ne kadar değişti?")]:
        if name in results:
            plot_heatmap(results[name]["rel"], title, out / f"heatmap_{name}.png")
    plot_layer_profile(results, out / "layer_profile.png")
    plot_spectra(results, out / "spectra.png")
    print("Yazıldı:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
