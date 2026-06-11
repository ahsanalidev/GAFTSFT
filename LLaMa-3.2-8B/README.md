# LLaMa-3.2-8B Experiments

## Model specification
- Base model: `unsloth/llama-3.2-8b-bnb-4bit`
- Family: LLaMA 3.2 Instruct
- Quantization: 4-bit (bnb) via Unsloth

## Expected runtime and memory
- Recommended GPU: A100 40GB (or equivalent) for comfortable training runs
- Typical VRAM footprint: high (8B model + optimizer states)
- Runtime: longest among the three model folders

## How to run
1. Open one of the notebooks in this folder:
   - `GradientAscentFTMethod.ipynb`
   - `IDKTuning.ipynb`
   - `PureGradientAscent.ipynb`
2. Execute cells in order in a GPU-backed environment.
3. Keep experiment seeds/parameters unchanged for fair comparison with other model folders.

## Output structure
- Training and intermediate artifacts: `LLaMa-3.2-8B/outputs/...`
- Evaluation artifacts and metrics: `LLaMa-3.2-8B/results/...`
