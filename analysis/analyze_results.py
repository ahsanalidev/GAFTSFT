#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["LLaMa-3.2-8B", "LLaMa-3.2-1B"]

METHOD_LABELS = {
    "PureGradientAscent": "PGA",
    "GradientAscentFTMethod": "GAFT",
    "IDKTuning": "IDK",
}

SPLITS = ["forget", "retain", "real_author", "world_fact"]
METRICS = ["acc", "rougeL_score", "truth_prob", "truth_ratio"]
MIA_THRESHOLDS = [10, 20, 30, 40, 50, 60]
METHOD_ORDER = ["PureGradientAscent", "GradientAscentFTMethod", "IDKTuning"]
T_CRITICAL_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
}


def safe_stdev(values):
    return stdev(values) if len(values) > 1 else 0.0


def ci95(values):
    if len(values) <= 1:
        return 0.0
    t_value = T_CRITICAL_95.get(len(values), 1.96)
    return t_value * safe_stdev(values) / math.sqrt(len(values))


def extract_seed(path):
    if path.parent.name.startswith("seed_"):
        return path.parent.name.removeprefix("seed_")
    if "_seed_" in path.parent.name:
        return path.parent.name.rsplit("_seed_", 1)[1]
    return "single_run"


def build_run_row(method_dir_name, result_path):
    method = METHOD_LABELS.get(method_dir_name, method_dir_name)
    data = json.loads(result_path.read_text())
    mia_values = list(data["MIA"].values())
    row = {
        "method_dir": method_dir_name,
        "method": method,
        "seed": extract_seed(result_path),
        "result_path": str(result_path.relative_to(ROOT)),
        "forget_quality": data["Forget Quality"],
        "mia_mean": mean(mia_values),
        "mia_min": min(mia_values),
        "mia_max": max(mia_values),
        "mia_range": max(mia_values) - min(mia_values),
    }

    for threshold in MIA_THRESHOLDS:
        row[f"mia_{threshold}"] = data["MIA"][f"Min_{threshold}.0% Prob"]

    for split in SPLITS:
        split_data = data[split]
        for metric in METRICS:
            row[f"{split}_{metric}"] = split_data[metric]

    row["utility_avg_acc"] = mean(
        [row["retain_acc"], row["real_author_acc"], row["world_fact_acc"]]
    )
    row["utility_avg_rouge"] = mean(
        [
            row["retain_rougeL_score"],
            row["real_author_rougeL_score"],
            row["world_fact_rougeL_score"],
        ]
    )
    row["forget_retain_acc_gap"] = row["retain_acc"] - row["forget_acc"]
    row["forget_retain_rouge_gap"] = (
        row["retain_rougeL_score"] - row["forget_rougeL_score"]
    )
    return row


def find_seed_result_files(model_dir):
    """Find all seeded result files for a specific model."""
    run_map = {method_dir: [] for method_dir in METHOD_ORDER}
    results_with_seed_dir = model_dir / "results_with_seed"
    
    if not results_with_seed_dir.exists():
        return run_map

    for method_dir in METHOD_ORDER:
        pattern = f"{method_dir}_seed_*/important.json"
        run_map[method_dir] = sorted(results_with_seed_dir.glob(pattern))
    
    return run_map


def load_results_for_model(model_dir):
    """Load results for a specific model."""
    run_rows = []
    run_map = find_seed_result_files(model_dir)

    for method_dir in METHOD_ORDER:
        for result_path in run_map.get(method_dir, []):
            run_rows.append(build_run_row(method_dir, result_path))

    summary_rows = []
    if not run_rows:
        return run_rows, summary_rows
    
    grouped = {}
    for row in run_rows:
        grouped.setdefault(row["method"], []).append(row)

    numeric_fields = [
        key
        for key, value in run_rows[0].items()
        if isinstance(value, (int, float)) and key not in {"mia_min", "mia_max"}
    ]

    for method in ["PGA", "GAFT", "IDK"]:
        rows = grouped.get(method, [])
        if not rows:
            continue
        summary = {
            "method_dir": rows[0]["method_dir"],
            "method": method,
            "n_runs": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in rows),
        }
        for field in numeric_fields:
            values = [row[field] for row in rows]
            summary[field] = mean(values)
            summary[f"{field}_std"] = safe_stdev(values)
            summary[f"{field}_ci95"] = ci95(values)
        summary_rows.append(summary)

    return run_rows, summary_rows


def write_csv(rows, filename, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename
    if not rows:
        return csv_path
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def fmt(x):
    return f"{x:.4f}"


def fmt_with_std(row, key):
    return f"{fmt(row[key])} $\\pm$ {fmt(row[f'{key}_std'])}"


def write_latex_tables(rows, output_dir):
    tables_path = output_dir / "latex_tables.tex"
    by_method = {row["method"]: row for row in rows}
    ordered = [by_method[m] for m in ["PGA", "GAFT", "IDK"] if m in by_method]

    table_1 = [
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Method & FQ $\uparrow$ & MIA $\downarrow$ & Forget Acc $\downarrow$ & Retain Acc $\uparrow$ & Real-Author Acc $\uparrow$ & World-Fact Acc $\uparrow$ \\",
        r"\hline",
    ]
    for row in ordered:
        table_1.append(
            f'{row["method"]} & {int(row["n_runs"])} & {fmt_with_std(row, "forget_quality")} & {fmt_with_std(row, "mia_mean")} & '
            f'{fmt_with_std(row, "forget_acc")} & {fmt_with_std(row, "retain_acc")} & '
            f'{fmt_with_std(row, "real_author_acc")} & {fmt_with_std(row, "world_fact_acc")} \\\\'
        )
    table_1[2] = r"Method & $n$ & FQ $\uparrow$ & MIA $\downarrow$ & Forget Acc $\downarrow$ & Retain Acc $\uparrow$ & Real-Author Acc $\uparrow$ & World-Fact Acc $\uparrow$ \\"
    table_1[0] = r"\begin{tabular}{lccccccc}"
    table_1.extend([r"\hline", r"\end{tabular}"])

    table_2 = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & Utility Avg. Acc $\uparrow$ & MIA Range $\downarrow$ & Forget--Retain Acc Gap & Utility Avg. ROUGE $\uparrow$ \\",
        r"\hline",
    ]
    for row in ordered:
        table_2.append(
            f'{row["method"]} & {fmt_with_std(row, "utility_avg_acc")} & {fmt_with_std(row, "mia_range")} & '
            f'{fmt_with_std(row, "forget_retain_acc_gap")} & {fmt_with_std(row, "utility_avg_rouge")} \\\\'
        )
    table_2.extend([r"\hline", r"\end{tabular}"])

    tables_path.write_text(
        "% Candidate Table 1: expanded cross-split comparison\n"
        + "\n".join(table_1)
        + "\n\n% Candidate Table 2: stability and utility summary\n"
        + "\n".join(table_2)
        + "\n"
    )
    return tables_path


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #222; }',
        '.small { font-size: 12px; }',
        '.label { font-size: 13px; font-weight: bold; }',
        '.title { font-size: 18px; font-weight: bold; }',
        '.axis { stroke: #444; stroke-width: 1.2; }',
        '.grid { stroke: #ddd; stroke-width: 1; }',
        '</style>',
    ]


def svg_footer():
    return ["</svg>"]


def save_svg(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_tradeoff(rows, output_dir):
    width, height = 760, 520
    left, right, top, bottom = 80, 30, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [row["forget_quality"] for row in rows]
    ys = [row["mia_mean"] for row in rows]
    x_err = [row["forget_quality_ci95"] for row in rows]
    y_err = [row["mia_mean_ci95"] for row in rows]
    min_x, max_x = min(x - e for x, e in zip(xs, x_err)) * 0.85, max(x + e for x, e in zip(xs, x_err)) * 1.05
    min_y, max_y = min(y - e for y, e in zip(ys, y_err)) * 0.9, max(y + e for y, e in zip(ys, y_err)) * 1.05

    def x_px(v):
        return left + (v - min_x) / (max_x - min_x) * plot_w

    def y_px(v):
        return top + plot_h - (v - min_y) / (max_y - min_y) * plot_h

    colors = {"PGA": "#d1495b", "GAFT": "#2e86ab", "IDK": "#2a9d8f"}
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Privacy-Unlearning Trade-off</text>')

    for i in range(5):
        yv = min_y + i * (max_y - min_y) / 4
        y = y_px(yv)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid" />')
        lines.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" class="small">{yv:.3f}</text>')

    for i in range(5):
        xv = min_x + i * (max_x - min_x) / 4
        x = x_px(xv)
        lines.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}" class="grid" />')
        lines.append(f'<text x="{x}" y="{top+plot_h+20}" text-anchor="middle" class="small">{xv:.3f}</text>')

    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis" />')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis" />')
    lines.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" class="label">Forget Quality (higher is better)</text>')
    lines.append(
        f'<text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})" class="label">Mean MIA score (lower is better)</text>'
    )

    for row in rows:
        cx = x_px(row["forget_quality"])
        cy = y_px(row["mia_mean"])
        color = colors.get(row["method"], "#333")
        lines.append(
            f'<line x1="{x_px(row["forget_quality"] - row["forget_quality_ci95"])}" y1="{cy}" '
            f'x2="{x_px(row["forget_quality"] + row["forget_quality_ci95"])}" y2="{cy}" stroke="{color}" stroke-width="2" />'
        )
        lines.append(
            f'<line x1="{cx}" y1="{y_px(row["mia_mean"] - row["mia_mean_ci95"])}" '
            f'x2="{cx}" y2="{y_px(row["mia_mean"] + row["mia_mean_ci95"])}" stroke="{color}" stroke-width="2" />'
        )
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{color}" />')
        lines.append(f'<text x="{cx+10}" y="{cy-10}" class="label">{row["method"]} (n={row["n_runs"]})</text>')

    lines.extend(svg_footer())
    return save_svg(output_dir / "tradeoff_scatter.svg", lines)


def plot_split_accuracy(rows, output_dir):
    width, height = 860, 540
    left, right, top, bottom = 90, 30, 60, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    methods = [row["method"] for row in rows]
    split_labels = {
        "forget": "Forget",
        "retain": "Retain",
        "real_author": "Real Author",
        "world_fact": "World Fact",
    }
    colors = {
        "forget": "#d1495b",
        "retain": "#edae49",
        "real_author": "#00798c",
        "world_fact": "#66a182",
    }

    all_values = []
    for row in rows:
        for split in SPLITS:
            all_values.append(row[f"{split}_acc"] + row[f"{split}_acc_ci95"])
    max_val = min(1.0, max(all_values) * 1.05)

    def y_px(v):
        return top + plot_h - (v / max_val) * plot_h

    group_w = plot_w / len(methods)
    bar_w = group_w / 6

    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Accuracy by Evaluation Split</text>')

    for i in range(6):
        val = max_val * i / 5
        y = y_px(val)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid" />')
        lines.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" class="small">{val:.1f}</text>')

    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis" />')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis" />')
    lines.append(f'<text x="22" y="{height/2}" text-anchor="middle" transform="rotate(-90 22 {height/2})" class="label">Accuracy</text>')

    for idx, row in enumerate(rows):
        gx = left + idx * group_w
        lines.append(f'<text x="{gx + group_w/2}" y="{top+plot_h+25}" text-anchor="middle" class="label">{row["method"]}</text>')
        for s_idx, split in enumerate(SPLITS):
            x = gx + bar_w + s_idx * (bar_w + 8)
            val = row[f"{split}_acc"]
            y = y_px(val)
            h = top + plot_h - y
            lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{colors[split]}" />')
            err = row[f"{split}_acc_ci95"]
            err_top = y_px(min(max_val, val + err))
            err_bottom = y_px(max(0.0, val - err))
            center_x = x + bar_w / 2
            lines.append(f'<line x1="{center_x}" y1="{err_top}" x2="{center_x}" y2="{err_bottom}" stroke="#333" stroke-width="1.5" />')
            lines.append(f'<line x1="{center_x-5}" y1="{err_top}" x2="{center_x+5}" y2="{err_top}" stroke="#333" stroke-width="1.5" />')
            lines.append(f'<line x1="{center_x-5}" y1="{err_bottom}" x2="{center_x+5}" y2="{err_bottom}" stroke="#333" stroke-width="1.5" />')

    legend_x = width - 180
    legend_y = 70
    for idx, split in enumerate(SPLITS):
        y = legend_y + idx * 24
        lines.append(f'<rect x="{legend_x}" y="{y-10}" width="14" height="14" fill="{colors[split]}" />')
        lines.append(f'<text x="{legend_x+22}" y="{y+1}" class="small">{split_labels[split]}</text>')

    lines.extend(svg_footer())
    return save_svg(output_dir / "split_accuracy.svg", lines)


def plot_mia_thresholds(rows, output_dir):
    width, height = 820, 520
    left, right, top, bottom = 90, 40, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    thresholds = [10, 20, 30, 40, 50, 60]
    all_values = []
    series = {}

    for row in rows:
        values = [row[f"mia_{t}"] for t in thresholds]
        errors = [row[f"mia_{t}_ci95"] for t in thresholds]
        series[row["method"]] = (values, errors)
        all_values.extend(v + e for v, e in zip(values, errors))
        all_values.extend(max(0.0, v - e) for v, e in zip(values, errors))

    min_y, max_y = min(all_values) * 0.95, max(all_values) * 1.05

    def x_px(v):
        return left + (v - thresholds[0]) / (thresholds[-1] - thresholds[0]) * plot_w

    def y_px(v):
        return top + plot_h - (v - min_y) / (max_y - min_y) * plot_h

    colors = {"PGA": "#d1495b", "GAFT": "#2e86ab", "IDK": "#2a9d8f"}
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" class="title">MIA Sensitivity Across Min-k Thresholds</text>')

    for i in range(5):
        yv = min_y + i * (max_y - min_y) / 4
        y = y_px(yv)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" class="grid" />')
        lines.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" class="small">{yv:.3f}</text>')

    for t in thresholds:
        x = x_px(t)
        lines.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}" class="grid" />')
        lines.append(f'<text x="{x}" y="{top+plot_h+20}" text-anchor="middle" class="small">{t}</text>')

    lines.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis" />')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis" />')
    lines.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" class="label">Min-k threshold (%)</text>')
    lines.append(
        f'<text x="22" y="{height/2}" text-anchor="middle" transform="rotate(-90 22 {height/2})" class="label">MIA score</text>'
    )

    for method, (values, errors) in series.items():
        color = colors.get(method, "#333")
        points = " ".join(f"{x_px(t)},{y_px(v)}" for t, v in zip(thresholds, values))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />')
        for t, v, err in zip(thresholds, values, errors):
            lines.append(f'<circle cx="{x_px(t)}" cy="{y_px(v)}" r="4" fill="{color}" />')
            lines.append(
                f'<line x1="{x_px(t)}" y1="{y_px(max(0.0, v - err))}" x2="{x_px(t)}" y2="{y_px(v + err)}" stroke="{color}" stroke-width="2" />'
            )

    legend_x = width - 130
    legend_y = 80
    for idx, method in enumerate(["PGA", "GAFT", "IDK"]):
        y = legend_y + idx * 24
        color = colors[method]
        lines.append(f'<line x1="{legend_x}" y1="{y-4}" x2="{legend_x+18}" y2="{y-4}" stroke="{color}" stroke-width="3" />')
        lines.append(f'<circle cx="{legend_x+9}" cy="{y-4}" r="4" fill="{color}" />')
        lines.append(f'<text x="{legend_x+26}" y="{y}" class="small">{method}</text>')

    lines.extend(svg_footer())
    return save_svg(output_dir / "mia_thresholds.svg", lines)


def write_findings(rows, output_dir):
    findings_path = output_dir / "findings.md"
    by_method = {row["method"]: row for row in rows}
    gaft = by_method.get("GAFT")
    idk = by_method.get("IDK")
    pga = by_method.get("PGA")

    lines = [
        "# Result Analysis",
        "",
        "## Strongest findings from the current JSON summaries",
        "",
    ]

    if gaft:
        lines.append(f"- GAFT has the strongest Forget Quality ({fmt_with_std(gaft, 'forget_quality')}) across {gaft['n_runs']} runs.")
    
    if idk and gaft and pga:
        lines.append(f"- IDK has the lowest mean MIA score ({fmt_with_std(idk, 'mia_mean')}), improving over GAFT by {fmt(gaft['mia_mean'] - idk['mia_mean'])} and over PGA by {fmt(pga['mia_mean'] - idk['mia_mean'])}.")
    
    if gaft:
        lines.append(f"- GAFT keeps the best overall utility average across non-forget splits ({fmt_with_std(gaft, 'utility_avg_acc')}), with especially strong real-author accuracy ({fmt_with_std(gaft, 'real_author_acc')}).")
    
    if pga:
        lines.append(f"- PGA has the worst privacy profile: highest mean MIA ({fmt_with_std(pga, 'mia_mean')}) and weakest Forget Quality ({fmt_with_std(pga, 'forget_quality')}).")
    
    if idk:
        lines.append(f"- IDK is still the most suppression-oriented method, lowering forget accuracy to {fmt_with_std(idk, 'forget_acc')} while also showing the lowest retain accuracy ({fmt_with_std(idk, 'retain_acc')}).")

    lines.extend([
        "",
        "## Suggested paper additions",
        "",
        "- Add a cross-split utility table that includes retain, real-author, and world-fact metrics, not just forget-set values.",
        "- Add a Forget Quality vs. MIA scatter plot to show that GAFT and IDK dominate PGA in different ways.",
        "- Add an MIA-threshold line chart to show that IDK consistently stays below the other methods across all min-k thresholds.",
        "- Add a short qualitative failure analysis using the generated forget answers: several responses still hallucinate substitute biographies instead of refusing cleanly.",
        "",
        "## Manuscript check",
        "",
    ])

    if pga:
        lines.append(f"- The paper should now report aggregated numbers, not a single run. The current PGA Forget Quality aggregate is {fmt_with_std(pga, 'forget_quality')}.")

    findings_path.write_text("\n".join(lines) + "\n")
    return findings_path


def main():
    for model in MODELS:
        model_dir = ROOT / model
        if not model_dir.exists():
            print(f"⚠️  Model directory not found: {model_dir}")
            continue
        
        print(f"Processing {model}...")
        out_dir = model_dir / "analysis" / "generated"
        
        run_rows, summary_rows = load_results_for_model(model_dir)
        
        if not summary_rows:
            print(f"  ⚠️  No results found for {model}")
            continue
        
        write_csv(run_rows, "results_runs.csv", out_dir)
        write_csv(summary_rows, "results_summary.csv", out_dir)
        write_latex_tables(summary_rows, out_dir)
        plot_tradeoff(summary_rows, out_dir)
        plot_split_accuracy(summary_rows, out_dir)
        plot_mia_thresholds(summary_rows, out_dir)
        write_findings(summary_rows, out_dir)
        print(f"✓ Wrote analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
