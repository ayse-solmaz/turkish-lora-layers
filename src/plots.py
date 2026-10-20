"""Grafikler ve özet tablolar.

Girdi: Kaggle'dan indirilen eval_bundle.zip'in açıldığı klasör
    <bundle>/train_logs/<deney>.json   eğitim logları (loss eğrileri)
    <bundle>/eval/<deney>.json         değerlendirme sonuçları

Çıktı: results/figures/*.png, results/summary.md, results/samples.md

Kullanım:
    python src/plots.py --bundle results/bundle_seed42
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Renkler: sabit sırada kategorik paletin ilk 3 slotu (renk körlüğü için doğrulanmış sıra)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

NAMES = {
    "base": "base (eğitimsiz)", "full_ft": "full FT", "lora_all_r16": "LoRA tümü r16",
    "lora_all_r4": "LoRA tümü r4", "lora_all_r64": "LoRA tümü r64",
    "lora_lower_r16": "LoRA alt yarı r16", "lora_upper_r16": "LoRA üst yarı r16",
    "lora_attn_r16": "LoRA attention r16", "lora_attn_r49": "LoRA attention r49",
    "lora_mlp_r16": "LoRA MLP r16",
}

# Her panel tek bir soruya cevap verir. Referans (LoRA tümü r16) her panelde aynı
# gri kesikli çizgi; bir deney birden fazla panelde görünüyorsa rengi de aynı kalır.
REF = "lora_all_r16"
PANELS = [
    ("Alt mı üst mü? (4.4M param)", [("lora_lower_r16", C1), ("lora_upper_r16", C2)]),
    ("Attention mı MLP mi? (~6.6M param)", [("lora_attn_r49", C1), ("lora_mlp_r16", C2)]),
    ("Rank (tüm modüller)", [("lora_all_r4", C1), ("lora_all_r64", C2)]),
    ("LoRA mı full FT mi?", [("lora_all_r64", C2), ("full_ft", C3)]),
]
REF_COLOR = "#8a8984"


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


def load(bundle: Path):
    train = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (bundle / "train_logs").glob("*.json")}
    ev = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (bundle / "eval").glob("*.json")}
    return train, ev


def plot_curves(train: dict, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True, facecolor=SURFACE)
    for ax, (title, exps) in zip(axes.flat, PANELS):
        style(ax, title)
        for exp, color in exps + [(REF, REF_COLOR)]:
            if exp not in train:
                continue
            steps, losses = zip(*train[exp]["val_loss"])
            ax.plot(steps, losses, color=color, linewidth=2, label=NAMES[exp] + (" (referans)" if exp == REF else ""),
                    linestyle="--" if exp == REF else "-")
            ax.annotate(f"{losses[-1]:.2f}", (steps[-1], losses[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
        ax.set_xlabel("optimizer adımı", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("val loss", fontsize=9, color=INK2)
    fig.suptitle("Doğrulama kaybı eğitim boyunca (düşük = iyi)", x=0.06, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def scatter_labeled(ax, xs, ys, labels):
    # eğitimsiz base gri: diğer noktaların "nereden geldiği"
    colors = [REF_COLOR if lab == NAMES["base"] else C1 for lab in labels]
    ax.scatter(xs, ys, s=60, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)


def plot_params(train: dict, ev: dict, out: Path):
    exps = [e for e in train if e in ev]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style(ax, "Türkçe test kaybı - eğitilen parametre sayısı")
    xs = [train[e]["trainable_params"] / 1e6 for e in exps]
    ys = [ev[e]["test_loss"] for e in exps]
    scatter_labeled(ax, xs, ys, [NAMES[e] for e in exps])
    if "base" in ev:
        ax.axhline(ev["base"]["test_loss"], color=INK2, linewidth=1, linestyle="--")
        ax.annotate("eğitimsiz base", (min(xs), ev["base"]["test_loss"]), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("eğitilen parametre (milyon, log ölçek)", fontsize=9, color=INK2)
    ax.set_ylabel("Türkçe test loss", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_forgetting(ev: dict, out: Path):
    exps = list(ev)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for ax, key, ylabel, title in [
        (axes[0], "en_ppl", "İngilizce perplexity (WikiText-2)", "Öğrenme - unutma: İngilizce metin"),
        (axes[1], "arc_easy", "ARC-Easy doğruluğu", "Öğrenme - unutma: genel bilgi"),
    ]:
        style(ax, title)
        xs = [ev[e]["test_loss"] for e in exps]
        ys = [ev[e][key] for e in exps]
        scatter_labeled(ax, xs, ys, [NAMES[e] for e in exps])
        ax.invert_xaxis()  # sağa gittikçe Türkçe daha iyi
        ax.set_xlabel("Türkçe test loss (sağa doğru daha iyi)", fontsize=9, color=INK2)
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_summary(train: dict, ev: dict, out: Path):
    order = sorted(ev, key=lambda e: ev[e]["test_loss"])
    lines = ["# Sonuç özeti", "",
             "| deney | param (M) | TR test loss | EN ppl | ARC-Easy | TR→TR | EN→EN | talimat | tekrar | bitirme |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in order:
        r, b = ev[e], ev[e]["behavior"]
        params = train[e]["trainable_params"] / 1e6 if e in train else 0
        lines.append(f"| {NAMES[e]} | {params:.2f} | {r['test_loss']:.3f} | {r['en_ppl']:.2f} | "
                     f"{r['arc_easy']:.3f} | {b['tr_answered_in_tr']:.0%} | {b['en_answered_in_en']:.0%} | "
                     f"{b['instruction_pass']:.0%} | {b['mean_repetition']:.2f} | {b['finished_rate']:.0%} |")
    lines += ["", "- **TR→TR / EN→EN:** Türkçe/İngilizce soruya aynı dilde cevap oranı (dil kayması)",
              "- **talimat:** madde sayısı, cümle sayısı, kelime sınırı kontrollerini geçme oranı",
              "- **tekrar:** tekrar eden 4-gram oranı (0 = hiç tekrar yok)",
              "- **bitirme:** cevabı 256 token dolmadan kendi bitirme oranı"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(ev: dict, out: Path):
    order = ["base", "lora_attn_r16", "lora_all_r16", "lora_all_r64", "full_ft"]
    order = [e for e in order if e in ev]
    gens = {e: {g["id"]: g for g in ev[e]["generations"]} for e in order}
    lines = ["# Örnek cevaplar", ""]
    for g in ev[order[0]]["generations"]:
        lines += [f"## {g['id']}", "", f"> {g['text']}", ""]
        for e in order:
            resp = gens[e][g["id"]]["response"].replace("\n", " ")
            lines.append(f"**{NAMES[e]}:** {resp[:500]}{'…' if len(resp) > 500 else ''}")
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    train, ev = load(Path(args.bundle))
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plot_curves(train, out / "figures" / "val_loss_curves.png")
    if ev:
        plot_params(train, ev, out / "figures" / "loss_vs_params.png")
        plot_forgetting(ev, out / "figures" / "learning_vs_forgetting.png")
        write_summary(train, ev, out / "summary.md")
        write_samples(ev, out / "samples.md")
    print("Yazıldı:", sorted(str(p) for p in out.rglob("*") if p.suffix in (".png", ".md")))


if __name__ == "__main__":
    main()
