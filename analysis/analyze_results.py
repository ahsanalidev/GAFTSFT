#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "analysis" / "generated"

METHOD_LABELS = {
    "PureGradientAscent": "PGA",
    "GradientAscentFTMethod": "GAFT",
    "IDKTuning": "IDK",
}

SPLITS = ["forget", "retain", "real_author", "world_fact"]
METRICS = ["acc", "rougeL_score", "truth_prob", "truth_ratio"]


def load_results():
    rows = []
    for important_path in sorted(RESULTS_DIR.glob("*/important.json")):
        method_dir = important_path.parent.name
        method = METHOD_LABELS.get(method_dir, method_dir)
        data = json.loads(important_path.read_text())
        mia_values = list(data["MIA"].values())

        row = {
            "method_dir": method_dir,
            "method": method,
            "forget_quality": data["Forget Quality"],
            "mia_mean": mean(mia_values),
            "mia_min": min(mia_values),
            "mia_max": max(mia_values),
            "mia_range": max(mia_values) - min(mia_values),
        }

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
        rows.append(row)
    return rows


def write_csv(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "results_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def fmt(x):
    return f"{x:.4f}"


def write_latex_tables(rows):
    tables_path = OUT_DIR / "latex_tables.tex"
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
            f'{row["method"]} & {fmt(row["forget_quality"])} & {fmt(row["mia_mean"])} & '
            f'{fmt(row["forget_acc"])} & {fmt(row["retain_acc"])} & '
            f'{fmt(row["real_author_acc"])} & {fmt(row["world_fact_acc"])} \\\\'
        )
    table_1.extend([r"\hline", r"\end{tabular}"])

    table_2 = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & Utility Avg. Acc $\uparrow$ & MIA Range $\downarrow$ & Forget--Retain Acc Gap & Utility Avg. ROUGE $\uparrow$ \\",
        r"\hline",
    ]
    for row in ordered:
        table_2.append(
            f'{row["method"]} & {fmt(row["utility_avg_acc"])} & {fmt(row["mia_range"])} & '
            f'{fmt(row["forget_retain_acc_gap"])} & {fmt(row["utility_avg_rouge"])} \\\\'
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


def plot_tradeoff(rows):
    width, height = 760, 520
    left, right, top, bottom = 80, 30, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [row["forget_quality"] for row in rows]
    ys = [row["mia_mean"] for row in rows]
    min_x, max_x = min(xs) * 0.85, max(xs) * 1.05
    min_y, max_y = min(ys) * 0.9, max(ys) * 1.05

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
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{color}" />')
        lines.append(f'<text x="{cx+10}" y="{cy-10}" class="label">{row["method"]}</text>')

    lines.extend(svg_footer())
    return save_svg(OUT_DIR / "tradeoff_scatter.svg", lines)


def plot_split_accuracy(rows):
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

    def y_px(v):
        return top + plot_h - v * plot_h

    group_w = plot_w / len(methods)
    bar_w = group_w / 6

    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" class="title">Accuracy by Evaluation Split</text>')

    for i in range(6):
        val = i / 5
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

    legend_x = width - 180
    legend_y = 70
    for idx, split in enumerate(SPLITS):
        y = legend_y + idx * 24
        lines.append(f'<rect x="{legend_x}" y="{y-10}" width="14" height="14" fill="{colors[split]}" />')
        lines.append(f'<text x="{legend_x+22}" y="{y+1}" class="small">{split_labels[split]}</text>')

    lines.extend(svg_footer())
    return save_svg(OUT_DIR / "split_accuracy.svg", lines)


def plot_mia_thresholds(rows):
    width, height = 820, 520
    left, right, top, bottom = 90, 40, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    thresholds = [10, 20, 30, 40, 50, 60]
    all_values = []
    series = {}

    for row in rows:
        important_path = RESULTS_DIR / row["method_dir"] / "important.json"
        data = json.loads(important_path.read_text())
        values = [data["MIA"][f"Min_{t}.0% Prob"] for t in thresholds]
        series[row["method"]] = values
        all_values.extend(values)

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

    for method, values in series.items():
        color = colors.get(method, "#333")
        points = " ".join(f"{x_px(t)},{y_px(v)}" for t, v in zip(thresholds, values))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />')
        for t, v in zip(thresholds, values):
            lines.append(f'<circle cx="{x_px(t)}" cy="{y_px(v)}" r="4" fill="{color}" />')

    legend_x = width - 130
    legend_y = 80
    for idx, method in enumerate(["PGA", "GAFT", "IDK"]):
        y = legend_y + idx * 24
        color = colors[method]
        lines.append(f'<line x1="{legend_x}" y1="{y-4}" x2="{legend_x+18}" y2="{y-4}" stroke="{color}" stroke-width="3" />')
        lines.append(f'<circle cx="{legend_x+9}" cy="{y-4}" r="4" fill="{color}" />')
        lines.append(f'<text x="{legend_x+26}" y="{y}" class="small">{method}</text>')

    lines.extend(svg_footer())
    return save_svg(OUT_DIR / "mia_thresholds.svg", lines)


def write_findings(rows):
    findings_path = OUT_DIR / "findings.md"
    by_method = {row["method"]: row for row in rows}
    gaft = by_method["GAFT"]
    idk = by_method["IDK"]
    pga = by_method["PGA"]

    lines = [
        "# Result Analysis",
        "",
        "## Strongest findings from the current JSON summaries",
        "",
        f"- GAFT has the strongest Forget Quality ({fmt(gaft['forget_quality'])}), far ahead of IDK ({fmt(idk['forget_quality'])}) and PGA ({fmt(pga['forget_quality'])}).",
        f"- IDK has the lowest mean MIA score ({fmt(idk['mia_mean'])}), improving over GAFT by {fmt(gaft['mia_mean'] - idk['mia_mean'])} and over PGA by {fmt(pga['mia_mean'] - idk['mia_mean'])}.",
        f"- GAFT keeps the best overall utility average across non-forget splits ({fmt(gaft['utility_avg_acc'])}), with especially strong real-author accuracy ({fmt(gaft['real_author_acc'])}).",
        f"- PGA has the worst privacy profile: highest mean MIA ({fmt(pga['mia_mean'])}) and weakest Forget Quality ({fmt(pga['forget_quality'])}).",
        f"- IDK is best described as a suppression-oriented method, because it lowers forget accuracy to {fmt(idk['forget_acc'])} and forget ROUGE to {fmt(idk['forget_rougeL_score'])}, but it also has the lowest retain accuracy ({fmt(idk['retain_acc'])}) and world-fact accuracy ({fmt(idk['world_fact_acc'])}).",
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
        f"- The current paper table reports PGA Forget Quality as 0.0299, but the latest `results/PureGradientAscent/important.json` stores {fmt(pga['forget_quality'])}. This looks like a stale number and should be reconciled before submission.",
    ]
    findings_path.write_text("\n".join(lines) + "\n")
    return findings_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    write_csv(rows)
    write_latex_tables(rows)
    plot_tradeoff(rows)
    plot_split_accuracy(rows)
    plot_mia_thresholds(rows)
    write_findings(rows)
    print(f"Wrote analysis artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
