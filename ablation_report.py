"""
Ablation report: comparison table + ablation table + visualizations.
Reads experiment results from output/tanks_and_temples/*/ and generates:
  - comparison_table.md   — cross-scene Baseline vs H1 vs H2 vs H1+H2
  - ablation_table.md     — per-component contribution breakdown
  - visualizations/
      render_comparison.png  — GT vs Baseline vs H1 vs H2 vs H1+H2 side-by-side
      psnr_per_view.png      — per-view PSNR boxplot
      metric_bars.png        — PSNR/SSIM/LPIPS bar chart
"""
import json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

OUTPUT_DIR = Path("output/tanks_and_temples")
REPORT_DIR = Path("ablation_report")
VIZ_DIR = REPORT_DIR / "visualizations"

# --- Experiment metadata ---
# (scene, exp_name) -> (label, group, iterations)
# Naming convention:
#   No suffix = Baseline (original Pixel-GS)
#   _H1 = Depth-normalized weight + remove SGF
#   _H2 = Geometric visibility floor (K=20)
#   _A  = All improvements combined (H1 + H2)
#   _D  = Depth-normalized weight only (7k ablation)
#   _S  = Remove SGF only (7k ablation)
#   _K5, _K50 = H2 K-value variants (7k ablation)
EXPERIMENTS = {
    # --- Auditorium: 30k ---
    ("Auditorium", "Auditorium"):    ("Baseline",     "baseline",  30000),
    ("Auditorium", "Auditorium_H1"): ("H1 (D+S)",     "h1",        30000),
    ("Auditorium", "Auditorium_H2"): ("H2 (K=20)",    "h2",        30000),
    ("Auditorium", "Auditorium_A"):  ("H1+H2 (All)",  "combined",  30000),
    # --- Auditorium: 7k ablations ---
    ("Auditorium", "Auditorium_D"):  ("D only",       "h1_abl",    7000),
    ("Auditorium", "Auditorium_S"):  ("S only",       "h1_abl",    7000),
    ("Auditorium", "Auditorium_K5"): ("H2 (K=5)",     "h2_abl",    7000),
    ("Auditorium", "Auditorium_K50"):("H2 (K=50)",    "h2_abl",    7000),
    # --- Family: 30k ---
    ("Family", "Family"):    ("Baseline",     "baseline",  30000),
    ("Family", "Family_H1"): ("H1 (D+S)",     "h1",        30000),
    ("Family", "Family_H2"): ("H2 (K=20)",    "h2",        30000),
    ("Family", "Family_A"):  ("H1+H2 (All)",  "combined",  30000),
    # --- Family: 7k ablations ---
    ("Family", "Family_D"):  ("D only",       "h1_abl",    7000),
    ("Family", "Family_S"):  ("S only",       "h1_abl",    7000),
    ("Family", "Family_K5"): ("H2 (K=5)",     "h2_abl",    7000),
    ("Family", "Family_K50"):("H2 (K=50)",    "h2_abl",    7000),
}

def load_results(exp_name):
    """Load results.json, returns (metrics_dict, iteration_key)."""
    path = OUTPUT_DIR / exp_name / "results.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    for key in data:
        return data[key], key
    return None, None

def load_per_view(exp_name):
    """Load per_view.json, returns per-view PSNR list."""
    path = OUTPUT_DIR / exp_name / "per_view.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for key in data:
        return list(data[key]["PSNR"].values())
    return None

def load_renders(exp_name):
    """Return list of (render_path, gt_path) pairs for an experiment."""
    test_dir = OUTPUT_DIR / exp_name / "test"
    if not test_dir.exists():
        return []
    for sub in test_dir.iterdir():
        if sub.is_dir() and sub.name.startswith("ours_"):
            renders_dir = sub / "renders"
            gt_dir = sub / "gt"
            if renders_dir.exists() and gt_dir.exists():
                renders = sorted(renders_dir.glob("*.png"))
                gts = sorted(gt_dir.glob("*.png"))
                return list(zip(renders, gts))
    return []

def ensure_dirs():
    REPORT_DIR.mkdir(exist_ok=True)
    VIZ_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. COMPARISON TABLE (30k-iter, Baseline vs H1 vs H2 vs H1+H2)
# ============================================================
def generate_comparison_table():
    scenes_order = [
        ("Auditorium", ["Auditorium", "Auditorium_H1", "Auditorium_H2", "Auditorium_A"]),
        ("Family",    ["Family",    "Family_H1",    "Family_H2",    "Family_A"]),
    ]
    method_labels = {
        "Auditorium": "Baseline", "Auditorium_H1": "H1 (D+S)", "Auditorium_H2": "H2 (K=20)", "Auditorium_A": "H1+H2 (All)",
        "Family": "Baseline", "Family_H1": "H1 (D+S)", "Family_H2": "H2 (K=20)", "Family_A": "H1+H2 (All)",
    }

    lines = []
    lines.append("# Comparison Table (30,000 iterations)")
    lines.append("")
    lines.append("| Scene | Method | PSNR↑ | SSIM↑ | LPIPS↓ | ΔPSNR | ΔSSIM | ΔLPIPS |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for scene, exp_names in scenes_order:
        # Baseline is the first entry
        base_m, _ = load_results(exp_names[0])
        if base_m is None:
            continue
        base_psnr = base_m["PSNR"]
        base_ssim = base_m["SSIM"]
        base_lpips = base_m["LPIPS"]

        first = True
        for exp_name in exp_names:
            metrics, _ = load_results(exp_name)
            if metrics is None:
                continue
            label = method_labels[exp_name]
            psnr = metrics["PSNR"]
            ssim = metrics["SSIM"]
            lpips = metrics["LPIPS"]

            if first:
                dpsnr = dssim = dlpips = "—"
            else:
                dpsnr = f"{psnr - base_psnr:+.2f}"
                dssim = f"{ssim - base_ssim:+.3f}"
                dlpips = f"{lpips - base_lpips:+.3f}"

            scene_cell = scene if first else ""
            lines.append(f"| {scene_cell} | {label} | {psnr:.2f} | {ssim:.3f} | {lpips:.3f} | {dpsnr} | {dssim} | {dlpips} |")
            first = False

    lines.append("")
    path = REPORT_DIR / "comparison_table.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {path}")
    return lines

# ============================================================
# 2. ABLATION TABLE
# ============================================================
def generate_ablation_table():
    lines = []
    lines.append("# Ablation Tables")
    lines.append("")

    # --- H1 ablation: D only, S only, D+S (H1) ---
    lines.append("## H1: Depth-Normalized Weight + Remove SGF")
    lines.append("")
    lines.append("| Scene | Iters | Ablation | PSNR↑ | SSIM↑ | LPIPS↓ | ΔPSNR |")
    lines.append("|---|---|---|---|---|---|---|")

    for scene in ["Auditorium", "Family"]:
        # 30k baseline
        base_m, _ = load_results(scene)
        base_psnr = base_m["PSNR"] if base_m else 0

        # 30k experiments: Baseline, H1
        for label, exp_name in [("Baseline", scene), ("H1 (D+S)", f"{scene}_H1")]:
            metrics, _ = load_results(exp_name)
            if metrics is None:
                continue
            dpsnr = "—" if label == "Baseline" else f"{metrics['PSNR'] - base_psnr:+.2f}"
            lines.append(f"| {scene} | 30k | {label} | {metrics['PSNR']:.2f} | {metrics['SSIM']:.3f} | {metrics['LPIPS']:.3f} | {dpsnr} |")

        # 7k ablations: S only, D only (compare both against S as weakest component)
        s_m, _ = load_results(f"{scene}_S")
        s_psnr = s_m["PSNR"] if s_m else 0
        for label, exp_name in [("S only", f"{scene}_S"), ("D only", f"{scene}_D")]:
            metrics, _ = load_results(exp_name)
            if metrics is None:
                continue
            if label == "S only":
                dpsnr = "—"
            else:
                dpsnr = f"{metrics['PSNR'] - s_psnr:+.2f} (vs S)"
            lines.append(f"| {scene} | 7k | {label} | {metrics['PSNR']:.2f} | {metrics['SSIM']:.3f} | {metrics['LPIPS']:.3f} | {dpsnr} |")

    lines.append("")

    # --- H2 K-scan ---
    lines.append("## H2: Geometric Visibility Floor — K-value Scan")
    lines.append("")
    lines.append("| Scene | K | PSNR↑ | SSIM↑ | LPIPS↓ | ΔPSNR | PSNR std↓ |")
    lines.append("|---|---|---|---|---|---|---|")

    for scene in ["Auditorium", "Family"]:
        base_m, _ = load_results(scene)
        base_psnr = base_m["PSNR"] if base_m else 0

        # Ordered by K value: 5 (7k), 20 (30k), 50 (7k)
        k_exps = [
            ("5", f"{scene}_K5", 7000),
            ("20", f"{scene}_H2", 30000),
            ("50", f"{scene}_K50", 7000),
        ]
        for k_label, exp_name, iters in k_exps:
            metrics, _ = load_results(exp_name)
            if metrics is None:
                continue
            pv = load_per_view(exp_name)
            std_str = f"{np.std(pv):.2f}" if pv else "—"
            dpsnr = f"{metrics['PSNR'] - base_psnr:+.2f}"
            lines.append(f"| {scene} | {k_label} ({iters//1000}k) | {metrics['PSNR']:.2f} | {metrics['SSIM']:.3f} | {metrics['LPIPS']:.3f} | {dpsnr} | {std_str} |")

        # Baseline row for reference
        base_m, _ = load_results(scene)
        if base_m:
            pv = load_per_view(scene)
            std_str = f"{np.std(pv):.2f}" if pv else "—"
            lines.append(f"| {scene} | ∞ (baseline) | {base_m['PSNR']:.2f} | {base_m['SSIM']:.3f} | {base_m['LPIPS']:.3f} | — | {std_str} |")

    lines.append("")
    path = REPORT_DIR / "ablation_table.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {path}")
    return lines

# ============================================================
# 3. VISUALIZATIONS
# ============================================================
def generate_visualizations():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        print("[!] matplotlib not installed. Run: pip install matplotlib")
        return

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.dpi': 150,
    })

    _render_comparison(plt, mpimg)
    _psnr_boxplot(plt)
    _metric_bars(plt)

    print(f"[[OK]] Visualizations saved to {VIZ_DIR}")

def _render_comparison(plt, mpimg):
    """Render comparison grid for each scene."""
    for scene in ["Auditorium", "Family"]:
        _render_scene_comparison(plt, mpimg, scene)

def _render_scene_comparison(plt, mpimg, scene):
    """Pick 2 test views, show GT + Baseline + H1 + H2 + H1+H2."""
    exp_base = scene  # e.g. "Auditorium" or "Family"
    exps = [
        ("GT",          exp_base,           True),
        ("Baseline",    exp_base,           False),
        ("H1 (D+S)",    f"{exp_base}_H1",   False),
        ("H2 (K=20)",   f"{exp_base}_H2",   False),
        ("H1+H2 (All)", f"{exp_base}_A",    False),
    ]

    all_pairs = {}
    for label, exp_name, _ in exps:
        all_pairs[label] = load_renders(exp_name)

    if not all_pairs["GT"]:
        print(f"[!] No rendered images found for {scene} comparison grid")
        return

    # Pick 2 views: one hard (low PSNR) and one easy (high PSNR)
    pv = load_per_view(exp_base)
    if pv:
        indices = np.argsort(pv)
        pick_indices = [indices[2], indices[-3]]
    else:
        pick_indices = [0, len(all_pairs["GT"]) // 2]

    n_views = len(pick_indices)
    n_methods = len(exps)
    fig, axes = plt.subplots(n_views, n_methods, figsize=(n_methods * 3, n_views * 3.2))

    if n_views == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(pick_indices):
        for col, (label, exp_name, is_gt) in enumerate(exps):
            ax = axes[row, col]
            pairs = all_pairs[label]
            if idx < len(pairs):
                img_path = pairs[idx][1] if is_gt else pairs[idx][0]
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.axis('off')
            psnr_str = ""
            if not is_gt:
                method_pv = load_per_view(exp_name)
                if method_pv and idx < len(method_pv):
                    psnr_str = f"\nPSNR={method_pv[idx]:.1f}"
            ax.set_title(f"{label}{psnr_str}", fontsize=9)

    fig.suptitle(f"{scene}: Render Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    filename = f"render_comparison_{scene}.png"
    fig.savefig(VIZ_DIR / filename, bbox_inches='tight')
    plt.close(fig)
    print(f"  [[OK]] {filename}")

def _psnr_boxplot(plt):
    """Per-view PSNR distribution: Baseline vs H1 vs H2 vs H1+H2."""
    exps = [
        ("Auditorium", "Auditorium",    "Baseline"),
        ("Auditorium", "Auditorium_H1", "H1 (D+S)"),
        ("Auditorium", "Auditorium_H2", "H2 (K=20)"),
        ("Auditorium", "Auditorium_A",  "H1+H2 (All)"),
        ("Family", "Family",    "Baseline"),
        ("Family", "Family_H1", "H1 (D+S)"),
        ("Family", "Family_H2", "H2 (K=20)"),
        ("Family", "Family_A",  "H1+H2 (All)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, scene in enumerate(["Auditorium", "Family"]):
        ax = axes[ax_idx]
        data = []
        labels = []
        for s, exp_name, label in exps:
            if s != scene:
                continue
            pv = load_per_view(exp_name)
            if pv:
                data.append(pv)
                labels.append(f"{label}\n(μ={np.mean(pv):.1f}, σ={np.std(pv):.2f})")

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        ax.set_title(f"{scene} Per-View PSNR Distribution")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("PSNR Variance Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "psnr_per_view.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  [[OK]] psnr_per_view.png")

def _metric_bars(plt):
    """Grouped bar chart: PSNR, SSIM, LPIPS for Baseline/H1/H2/H1+H2 on both scenes."""
    scenes = ["Auditorium", "Family"]
    methods = ["Baseline", "H1 (D+S)", "H2 (K=20)", "H1+H2 (All)"]
    exp_map = {
        ("Auditorium", "Baseline"):    "Auditorium",
        ("Auditorium", "H1 (D+S)"):    "Auditorium_H1",
        ("Auditorium", "H2 (K=20)"):   "Auditorium_H2",
        ("Auditorium", "H1+H2 (All)"): "Auditorium_A",
        ("Family", "Baseline"):    "Family",
        ("Family", "H1 (D+S)"):    "Family_H1",
        ("Family", "H2 (K=20)"):   "Family_H2",
        ("Family", "H1+H2 (All)"): "Family_A",
    }

    metrics_names = ["PSNR", "SSIM", "LPIPS"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(scenes))
    width = 0.18
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd']

    for m_idx, metric in enumerate(metrics_names):
        ax = axes[m_idx]
        for j, method in enumerate(methods):
            values = []
            for scene in scenes:
                exp_name = exp_map[(scene, method)]
                m, _ = load_results(exp_name)
                values.append(m[metric] if m else 0)
            offset = (j - 1.5) * width
            bars = ax.bar(x + offset, values, width, label=method, color=colors[j], alpha=0.85)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01 * max(values),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=6.5)

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        if m_idx == 0:
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Metric Comparison: Baseline vs H1 vs H2 vs H1+H2", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "metric_bars.png", bbox_inches='tight')
    plt.close(fig)
    print(f"  [[OK]] metric_bars.png")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    ensure_dirs()
    print("=" * 60)
    print("Ablation Report Generator")
    print("=" * 60)

    print("\n[1/3] Comparison Table...")
    generate_comparison_table()

    print("\n[2/3] Ablation Table...")
    generate_ablation_table()

    print("\n[3/3] Visualizations...")
    generate_visualizations()

    print(f"\n{'=' * 60}")
    print(f"Done. Output: {REPORT_DIR.resolve()}")
    print(f"{'=' * 60}")
