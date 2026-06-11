#!/usr/bin/env python3
"""Cross-model comparison of unlearning methods across LLaMA-3.2 model scales.

Reads the per-model ``results_summary.csv`` files produced by
``analyze_results.py`` and renders a single 2x2 panel SVG comparing the 8B and
1B models on the four evaluation axes used in the thesis:

  A. Forget Quality (higher is better)
  B. Mean MIA Min-k% score (lower is better)
  C. Utility-average accuracy across non-forget splits (higher is better)
  D. Seed stability: standard deviation of Forget Quality across seeds (lower
     is better)

The chart is hand-rolled SVG (no plotting dependencies) to match the rest of
the analysis pipeline. A copy is written into each model's
``analysis/generated`` directory so the figure travels with either model.
"""
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["LLaMa-3.2-8B", "LLaMa-3.2-1B"]
MODEL_LABELS = {"LLaMa-3.2-8B": "8B", "LLaMa-3.2-1B": "1B"}
MODEL_COLORS = {"LLaMa-3.2-8B": "#2e86ab", "LLaMa-3.2-1B": "#e07a1f"}
METHOD_ORDER = ["PGA", "GAFT", "IDK"]


def load_summary(model):
    path = ROOT / model / "analysis" / "generated" / "results_summary.csv"
    if not path.exists():
        return None
    rows = {row["method"]: row for row in csv.DictReader(path.open())}
    return rows


def getf(rows, method, field):
    value = rows.get(method, {}).get(field, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def svg_panel(x0, y0, w, h, title, arrow, value_fn, err_fn, summaries, fmt="{:.3f}"):
    """Render one grouped-bar panel (3 methods x N models) into SVG lines."""
    pad_left, pad_right, pad_top, pad_bottom = 52, 14, 40, 42
    plot_x = x0 + pad_left
    plot_y = y0 + pad_top
    plot_w = w - pad_left - pad_right
    plot_h = h - pad_top - pad_bottom

    available = [m for m in MODELS if summaries.get(m)]
    all_tops = []
    for method in METHOD_ORDER:
        for model in available:
            all_tops.append(value_fn(summaries[model], method) + err_fn(summaries[model], method))
    max_val = max(all_tops) if all_tops else 1.0
    max_val = max_val * 1.18 if max_val > 0 else 1.0

    def y_px(v):
        return plot_y + plot_h - (v / max_val) * plot_h

    lines = []
    lines.append(
        f'<text x="{x0 + w/2}" y="{y0 + 22}" text-anchor="middle" class="ptitle">{title} {arrow}</text>'
    )

    # gridlines + y ticks
    for i in range(5):
        v = max_val * i / 4
        y = y_px(v)
        lines.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" class="grid" />')
        lines.append(f'<text x="{plot_x - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt.format(v)}</text>')

    lines.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" class="axis" />')
    lines.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" class="axis" />')

    group_w = plot_w / len(METHOD_ORDER)
    n_models = max(len(available), 1)
    bar_w = group_w / (n_models + 1.6)
    for g_idx, method in enumerate(METHOD_ORDER):
        gx = plot_x + g_idx * group_w
        # center the bars within the group slot
        block_w = n_models * bar_w + (n_models - 1) * (bar_w * 0.25)
        start = gx + (group_w - block_w) / 2
        for m_idx, model in enumerate(available):
            val = value_fn(summaries[model], method)
            err = err_fn(summaries[model], method)
            bx = start + m_idx * (bar_w + bar_w * 0.25)
            by = y_px(val)
            bh = plot_y + plot_h - by
            color = MODEL_COLORS[model]
            lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" rx="1.5" />')
            # value label
            lines.append(
                f'<text x="{bx + bar_w/2:.1f}" y="{by - 13:.1f}" text-anchor="middle" class="bval">{fmt.format(val)}</text>'
            )
            # error bar
            if err > 0:
                cx = bx + bar_w / 2
                top = y_px(val + err)
                bot = y_px(max(0.0, val - err))
                lines.append(f'<line x1="{cx:.1f}" y1="{top:.1f}" x2="{cx:.1f}" y2="{bot:.1f}" stroke="#333" stroke-width="1.3" />')
                lines.append(f'<line x1="{cx-4:.1f}" y1="{top:.1f}" x2="{cx+4:.1f}" y2="{top:.1f}" stroke="#333" stroke-width="1.3" />')
                lines.append(f'<line x1="{cx-4:.1f}" y1="{bot:.1f}" x2="{cx+4:.1f}" y2="{bot:.1f}" stroke="#333" stroke-width="1.3" />')
        lines.append(
            f'<text x="{gx + group_w/2:.1f}" y="{plot_y + plot_h + 24:.1f}" text-anchor="middle" class="mlabel">{method}</text>'
        )
    return lines


def build_svg(summaries):
    width, height = 1000, 760
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #222; }',
        '.title { font-size: 22px; font-weight: bold; }',
        '.subtitle { font-size: 13px; fill: #555; }',
        '.ptitle { font-size: 16px; font-weight: bold; }',
        '.mlabel { font-size: 14px; font-weight: bold; }',
        '.tick { font-size: 11px; fill: #444; }',
        '.bval { font-size: 11px; fill: #222; }',
        '.legend { font-size: 14px; }',
        '.axis { stroke: #444; stroke-width: 1.2; }',
        '.grid { stroke: #e3e3e3; stroke-width: 1; }',
        '</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Cross-Model Comparison: LLaMA-3.2-8B vs LLaMA-3.2-1B</text>',
        f'<text x="{width/2}" y="50" text-anchor="middle" class="subtitle">Three unlearning methods, mean over seeds 3407/3408/3409 (error bars: std. dev. across seeds)</text>',
    ]

    # legend
    lx = width / 2 - 110
    ly = 68
    for model in MODELS:
        if not summaries.get(model):
            continue
        lines.append(f'<rect x="{lx}" y="{ly - 11}" width="16" height="14" fill="{MODEL_COLORS[model]}" rx="2" />')
        lines.append(f'<text x="{lx + 22}" y="{ly + 1}" class="legend">LLaMA-3.2-{MODEL_LABELS[model]}</text>')
        lx += 150

    panel_w = (width - 40) / 2
    panel_h = (height - 100) / 2
    top = 90
    panels = [
        (20, top, "Forget Quality", "↑",
         lambda r, m: getf(r, m, "forget_quality"),
         lambda r, m: getf(r, m, "forget_quality_std"), "{:.2f}"),
        (20 + panel_w, top, "Mean MIA (Min-k%)", "↓",
         lambda r, m: getf(r, m, "mia_mean"),
         lambda r, m: getf(r, m, "mia_mean_std"), "{:.2f}"),
        (20, top + panel_h, "Utility Avg. Accuracy", "↑",
         lambda r, m: getf(r, m, "utility_avg_acc"),
         lambda r, m: getf(r, m, "utility_avg_acc_std"), "{:.2f}"),
        (20 + panel_w, top + panel_h, "Seed Stability (Forget Quality σ)", "↓",
         lambda r, m: getf(r, m, "forget_quality_std"),
         lambda r, m: 0.0, "{:.3f}"),
    ]
    for x0, y0, title, arrow, vfn, efn, fmt in panels:
        lines.extend(svg_panel(x0, y0, panel_w, panel_h, title, arrow, vfn, efn, summaries, fmt=fmt))

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


IDK_CATEGORY_ORDER = [
    "clean_abstention",
    "partial_abstention",
    "empty_or_no_answer",
    "original_memory_leakage",
    "hallucinated_substitute",
    "unrelated_drift",
]
IDK_CATEGORY_LABELS = {
    "clean_abstention": "Clean abstention",
    "partial_abstention": "Partial abstention",
    "empty_or_no_answer": "Empty / no answer",
    "original_memory_leakage": "Original-memory leakage",
    "hallucinated_substitute": "Hallucinated substitute",
    "unrelated_drift": "Unrelated drift",
}
IDK_CATEGORY_COLORS = {
    "clean_abstention": "#2a9d8f",
    "partial_abstention": "#8ecae6",
    "empty_or_no_answer": "#adb5bd",
    "original_memory_leakage": "#6a4c93",
    "hallucinated_substitute": "#d1495b",
    "unrelated_drift": "#edae49",
}


def load_idk_distribution(model):
    """Return {category: percentage} from the IDK behavioral classification CSV."""
    path = ROOT / model / "analysis" / "generated" / "idk_behavior_full_classification.csv"
    if not path.exists():
        return None
    counts = {cat: 0 for cat in IDK_CATEGORY_ORDER}
    total = 0
    for row in csv.DictReader(path.open()):
        cat = row.get("category")
        if cat in counts:
            counts[cat] += 1
            total += 1
    if not total:
        return None
    return {cat: counts[cat] / total * 100.0 for cat in IDK_CATEGORY_ORDER}


def build_idk_behavior_svg(distributions):
    """Stacked horizontal bars of IDK forget-set behavior, one row per model."""
    width, height = 1000, 420
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #222; }',
        '.title { font-size: 22px; font-weight: bold; }',
        '.subtitle { font-size: 13px; fill: #555; }',
        '.mlabel { font-size: 15px; font-weight: bold; }',
        '.seg { font-size: 12px; fill: #fff; font-weight: bold; }',
        '.legend { font-size: 13px; }',
        '</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">IDK Forget-Set Behavior by Model Scale</text>',
        f'<text x="{width/2}" y="50" text-anchor="middle" class="subtitle">Heuristic classification of generated forget answers, pooled over seeds 3407/3408/3409 (n=1200 each)</text>',
    ]

    bar_left, bar_right = 70, 40
    bar_w = width - bar_left - bar_right
    bar_h = 70
    row_gap = 34
    top = 90
    available = [m for m in MODELS if distributions.get(m)]
    for idx, model in enumerate(available):
        y = top + idx * (bar_h + row_gap)
        dist = distributions[model]
        lines.append(f'<text x="{bar_left}" y="{y - 8}" class="mlabel">LLaMA-3.2-{MODEL_LABELS[model]}</text>')
        x = bar_left
        for cat in IDK_CATEGORY_ORDER:
            pct = dist[cat]
            seg_w = bar_w * pct / 100.0
            if seg_w <= 0:
                continue
            color = IDK_CATEGORY_COLORS[cat]
            lines.append(f'<rect x="{x:.1f}" y="{y}" width="{seg_w:.1f}" height="{bar_h}" fill="{color}" />')
            if seg_w > 34:
                lines.append(
                    f'<text x="{x + seg_w/2:.1f}" y="{y + bar_h/2 + 4:.1f}" text-anchor="middle" class="seg">{pct:.0f}%</text>'
                )
            x += seg_w
        lines.append(f'<rect x="{bar_left}" y="{y}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#444" stroke-width="1" />')

    # legend
    legend_y = top + len(available) * (bar_h + row_gap) + 6
    col_x = [70, 270, 470, 690]
    for i, cat in enumerate(IDK_CATEGORY_ORDER):
        cx = col_x[i % 4] if i < 4 else [70, 270, 470, 690][i - 4]
        ly = legend_y + (0 if i < 4 else 24)
        lines.append(f'<rect x="{cx}" y="{ly - 11}" width="14" height="14" fill="{IDK_CATEGORY_COLORS[cat]}" />')
        lines.append(f'<text x="{cx + 20}" y="{ly + 1}" class="legend">{IDK_CATEGORY_LABELS[cat]}</text>')

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main():
    summaries = {model: load_summary(model) for model in MODELS}
    available = [m for m in MODELS if summaries.get(m)]
    if len(available) < 2:
        print(f"⚠️  Need both model summaries; found: {available}. Run analyze_results.py first.")
        return

    svg = build_svg(summaries)
    distributions = {model: load_idk_distribution(model) for model in MODELS}
    idk_available = [m for m in MODELS if distributions.get(m)]
    idk_svg = build_idk_behavior_svg(distributions) if len(idk_available) >= 1 else None

    for model in available:
        out_dir = ROOT / model / "analysis" / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cross_model_comparison.svg"
        out_path.write_text(svg)
        print(f"✓ Wrote {out_path}")
        if idk_svg is not None:
            idk_path = out_dir / "cross_model_idk_behavior.svg"
            idk_path.write_text(idk_svg)
            print(f"✓ Wrote {idk_path}")


if __name__ == "__main__":
    main()
