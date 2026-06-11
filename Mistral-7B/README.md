# Mistral-7B Experiments

## Model specification
- Base model: `unsloth/Mistral-7B-v0.3-bnb-4bit`
- Family: Mistral 7B
- Quantization: 4-bit (bnb) via Unsloth

## Expected runtime and memory
- Recommended GPU: A100 40GB (or equivalent); lower than LLaMA-3.2-8B in some runs
- Typical VRAM footprint: medium-to-high
- Runtime: typically between LLaMA-3.2-1B and LLaMA-3.2-8B

## How to run
1. Open one of the notebooks in this folder:
   - `GradientAscentFTMethod.ipynb`
   - `IDKTuning.ipynb`
   - `PureGradientAscent.ipynb`
2. Execute cells in order in a GPU-backed environment.
3. Keep experiment seeds/parameters unchanged for fair comparison with other model folders.
4. If tokenizer behavior differs for Mistral, keep pad token fallback (`eos_token`) enabled as already configured in notebooks.

## Output structure
- Training and intermediate artifacts: `Mistral-7B/outputs/...`
- Evaluation artifacts and metrics: `Mistral-7B/results/...`
