#!/usr/bin/env python3
"""Render the generated SVG figures to PNGs used by the LaTeX thesis.

The analysis pipeline emits SVGs; the thesis is compiled with pdflatex, which
consumes PNG/PDF. This script converts the per-model and cross-model SVGs into
PNGs under ``ms-thesis/images/`` with stable, disambiguated names:

  <fig>.png       -> LLaMA-3.2-8B figure (used in the main results chapter)
  <fig>_1b.png    -> LLaMA-3.2-1B figure (used in the appendix)
  cross_model_*.png

Requires cairosvg (``pip install cairosvg``).
"""
from pathlib import Path

import cairosvg


ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "ms-thesis" / "images"
SCALE = 2.4  # crisp for print

PER_MODEL_FIGS = ["tradeoff_scatter", "split_accuracy", "mia_thresholds"]
CROSS_MODEL_FIGS = ["cross_model_comparison", "cross_model_idk_behavior"]


def gen_dir(model):
    return ROOT / model / "analysis" / "generated"


def convert(src, dst):
    if not src.exists():
        print(f"⚠️  missing {src}")
        return
    cairosvg.svg2png(url=str(src), write_to=str(dst), scale=SCALE)
    print(f"✓ {dst.relative_to(ROOT)}")


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    # 8B figures keep the bare name (referenced by the main results chapter).
    for fig in PER_MODEL_FIGS:
        convert(gen_dir("LLaMa-3.2-8B") / f"{fig}.svg", IMG / f"{fig}.png")
    # 1B figures get the _1b suffix (referenced by the appendix).
    for fig in PER_MODEL_FIGS:
        convert(gen_dir("LLaMa-3.2-1B") / f"{fig}.svg", IMG / f"{fig}_1b.png")
    # Cross-model figures are identical in both folders; take the 8B copy.
    for fig in CROSS_MODEL_FIGS:
        convert(gen_dir("LLaMa-3.2-8B") / f"{fig}.svg", IMG / f"{fig}.png")


if __name__ == "__main__":
    main()
