# LLaMa-3.2-1B Experiments

## Model specification
- Base model: `unsloth/Llama-3.2-1B-Instruct`
- Family: LLaMA 3.2 Instruct
- Quantization/runtime path: Unsloth (4-bit loading available in Unsloth stack)

## Expected runtime and memory
- Recommended GPU: T4/A10/A100 (lower VRAM requirement than 8B/7B variants)
- Typical VRAM footprint: low-to-medium relative to other folders
- Runtime: fastest among the three model folders

## How to run
1. Open one of the notebooks in this folder:
   - `GradientAscentFTMethod.ipynb`
   - `IDKTuning.ipynb`
   - `PureGradientAscent.ipynb`
2. Execute cells in order in a GPU-backed environment.
3. Keep experiment seeds/parameters unchanged for fair comparison with other model folders.

## Output structure
- Training and intermediate artifacts: `LLaMa-3.2-1B/outputs/...`
- Evaluation artifacts and metrics: `LLaMa-3.2-1B/results/...`
