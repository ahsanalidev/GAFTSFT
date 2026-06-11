# Cross-Model Comparison: LLaMA-3.2-8B vs LLaMA-3.2-1B

**Thesis:** *Auditing Privacy in Machine Unlearning using Poisoning*
**Methods compared:** PGA (Pure Gradient Ascent), GAFT (Gradient Ascent with a retain/fine-tuning constraint), IDK (I-Don't-Know refusal tuning)
**Protocol:** identical two-stage *memorize-then-unlearn* pipeline run entirely in 4-bit (NF4) + LoRA, TOFU `forget01`, three seeds (3407 / 3408 / 3409) per configuration, at each model scale.
**Evaluation axes:** Forgetting efficacy (Forget Quality, FQ), Privacy (Min-*k*% MIA), Utility preservation (per-split accuracy), Stability (std. dev. across seeds).

All numbers below are means over the three seeds (± std. dev.). They are produced by `analysis/analyze_results.py`, `analysis/idk_behavior_analysis.py`, and `analysis/cross_model_comparison.py`, which read each model's `results/` directory and write per-model artifacts to `<model>/analysis/generated/`.

---

## 1. Headline metrics

| Axis | Method | 8B | 1B | Δ (1B − 8B) |
|------|--------|-----|-----|-------------|
| **Forget Quality** ↑ | PGA  | 0.0392 ± 0.0046 | 0.0408 ± 0.0023 | +0.0016 |
|                      | GAFT | **0.5427 ± 0.0334** | **0.5610 ± 0.0381** | +0.0183 |
|                      | IDK  | 0.1037 ± 0.0441 | 0.1052 ± 0.0406 | +0.0015 |
| **Mean MIA** ↓ | PGA  | 0.6219 ± 0.0130 | 0.6420 ± 0.0167 | +0.0201 |
|                | GAFT | 0.4418 ± 0.0087 | 0.4516 ± 0.0092 | +0.0098 |
|                | IDK  | **0.4171 ± 0.0070** | **0.4251 ± 0.0076** | +0.0080 |
| **Utility Avg. Acc** ↑ | PGA  | 0.7901 ± 0.0047 | **0.7135 ± 0.0051** | −0.0766 |
|                        | GAFT | **0.7912 ± 0.0107** | 0.7112 ± 0.0120 | −0.0800 |
|                        | IDK  | 0.7799 ± 0.0032 | 0.7056 ± 0.0009 | −0.0743 |
| **Forget Acc** ↓ | PGA  | 0.6183 ± 0.0038 | 0.5889 ± 0.0069 | −0.0294 |
|                  | GAFT | 0.5700 ± 0.0025 | 0.5412 ± 0.0079 | −0.0288 |
|                  | IDK  | **0.5617 ± 0.0063** | **0.5369 ± 0.0058** | −0.0248 |

(Bold = best method *within that model* for that axis.)

---

## 2. Key findings — LLaMA-3.2-8B

- **GAFT is the most balanced method.** It wins distributional forgetting (FQ = 0.5427, >5× IDK and >13× PGA), ties/leads utility (0.7912 avg acc), and sits in the low-leakage region (MIA = 0.4418).
- **IDK has the best raw privacy** (MIA = 0.4171, lowest at every Min-*k*% threshold) and the most aggressive suppression (lowest forget acc, lowest truth prob), **but the behavioral audit disqualifies this as trustworthy forgetting** (see §4).
- **PGA is dominated everywhere:** near-zero FQ (0.0392) *and* the highest leakage (MIA = 0.6219, well above chance at every threshold).
- Utility is high and well-preserved: real-author ≈ 0.92, world-fact ≈ 0.88 across all methods.

## 3. Key findings — LLaMA-3.2-1B

- **The method ranking is identical to 8B.** GAFT wins FQ (0.5610), IDK wins MIA (0.4251), PGA is worst on both. The privacy-utility trade-off geometry (GAFT far-right, IDK bottom, PGA stranded upper-left) reproduces at 1B.
- **GAFT's FQ is marginally *higher* at 1B** (0.5610 vs 0.5427); PGA and IDK FQ are essentially unchanged. Forgetting efficacy does not degrade with scale.
- **Privacy leakage is marginally worse at 1B** for every method (MIA up ~0.008–0.020), i.e. the smaller model is slightly more attackable, but PGA's large gap above GAFT/IDK persists and GAFT/IDK stay near the no-signal region.
- **Utility is substantially lower in absolute terms** (avg acc ≈ 0.71 vs ≈ 0.79; real-author 0.92→0.82, world-fact 0.88→0.79). This is a property of the **smaller base model's lower capability**, not of unlearning: the three methods stay within ~0.008 of each other at 1B, the same tight spread as at 8B, so the *marginal* utility cost of unlearning is similar at both scales.

---

## 4. IDK behavioral audit across scales (the central insight)

Heuristic classification of 1,200 IDK forget-set generations per model (3 seeds × 400):

| Category | 8B | 1B |
|----------|----:|----:|
| Clean abstention | 0.0% | 18.8% |
| Partial abstention | 0.0% | 0.2% |
| Empty / no answer | 0.1% | 0.5% |
| **Hallucinated substitute** | **86.1%** | **50.9%** |
| Unrelated drift | 5.6% | 27.0% |
| Memorized leakage (proxy) | 8.2% | 2.6% |

**The "low overlap ≠ clean abstention" finding holds at both scales.** IDK was trained to answer forget questions with *"I do not have that information,"* yet **hallucinated substitution is the plurality behavior at both sizes** (86.1% at 8B, 50.9% at 1B), and clean abstention never becomes the dominant mode.

**Scale nuance in *how* it fails:**
- The **8B** model is fluent enough to fail "cleanly looking" — it almost always manufactures a confident, plausible biography (86.1% hallucination, **0% genuine refusal**), and paraphrases the original fact often enough to leak it in 8.2% of cases.
- The **1B** model fails in a **noisier, more fragmented** way: it produces *more* genuine refusals (18.8% vs 0%) and far more degenerate/off-topic drift (27.0% vs 5.6%), and leaks less by paraphrase (2.6%) — but fabrication is still the single largest category. The smaller model is simply less capable of fluent invention, so the failure spreads across abstention, drift, and shorter fabrications.
- Per-seed stability of the dominant mode: hallucination share was **84.5–88.5%** per seed at 8B and **46.8–53.5%** per seed at 1B — stable within each scale.

**Takeaway:** at neither scale does IDK deliver the clean refusal its training objective implies; the apparent forgetting is produced by answer *replacement*, not removal. Smaller scale does not fix this — it changes the texture of the failure, not its existence.

---

## 5. Stability across seeds

- Between-method gaps dwarf seed-level variation at **both** scales, so all rankings are robust to training randomness.
- **IDK's Forget Quality remains the single most volatile number at both scales** (std ≈ mean: 0.0441 vs 0.1037 at 8B; 0.0406 vs 0.1052 at 1B). This corroborates the audit at both sizes: a fabrication-driven mechanism produces erratic distributional forgetting regardless of scale.
- 1B variances are marginally larger across the board (a smaller model is slightly noisier), but never enough to disturb a ranking. MIA std stays ≈ 0.01 or below at both scales.

---

## 6. Cross-model observations & do the 8B conclusions extend to 1B?

| Conclusion (established on 8B) | Holds on 1B? |
|---|---|
| Objective choice dominates the outcome (FQ spans ~0→0.55; MIA spans chance→0.64) | **Yes** |
| Method ranking PGA ≺ {GAFT, IDK}, with GAFT and IDK dominating PGA on different axes | **Yes** |
| GAFT is the most balanced / most dependable method | **Yes** (PGA edges utility by 0.002 at 1B, but GAFT still wins FQ + low MIA + stability) |
| IDK has best raw MIA but suppresses by fabrication, not abstention | **Yes** (hallucination is the plurality at 1B too) |
| PGA is dominated everywhere (worst FQ + worst leakage) | **Yes** |
| IDK Forget Quality is the least reliable single number | **Yes** |
| Forgetting/privacy rankings survive within the 4-bit quantized pipeline | **Yes** — *quantization and quantized-unlearning behavior is consistent across model scales* |

**Scale-specific insights:**
1. **Forgetting efficacy is scale-robust** (FQ ordering and magnitudes preserved; GAFT marginally better at 1B).
2. **Privacy is marginally scale-sensitive**: the smaller model leaks slightly more (higher MIA for every method), consistent with a smaller, more memorization-prone-per-capacity model, but the qualitative ordering is unchanged.
3. **Utility is the most scale-sensitive axis** — but the drop (~8 points) is a base-model capability effect, not an unlearning effect; the *relative* method ordering and the small inter-method spread are preserved.
4. **IDK's failure mode shifts with scale** (fluent fabrication at 8B → fragmented abstention/drift/fabrication at 1B) while the core verdict (no clean abstention; fabrication dominant) is invariant.

**Bottom line:** every substantive conclusion from the 8B study extends to 1B. Model scale changes absolute utility and the *texture* of IDK's failure, but it does not change which method to trust (GAFT), which metric to distrust (IDK's low overlap / FQ), or the lesson that unlearning behavior — including under quantization — must be validated at the generation level.

---

## 7. Generated artifacts

Per model (`LLaMa-3.2-8B/analysis/generated/`, `LLaMa-3.2-1B/analysis/generated/`):
- `results_runs.csv`, `results_summary.csv` — per-seed and aggregated metrics
- `latex_tables.tex`, `findings.md`
- `tradeoff_scatter.svg`, `split_accuracy.svg`, `mia_thresholds.svg`
- `idk_behavior_full_classification.csv`, `idk_behavior_samples.csv`
- `cross_model_comparison.svg` — 4-panel FQ / MIA / Utility / seed-stability comparison
- `cross_model_idk_behavior.svg` — IDK behavior breakdown, 8B vs 1B

Thesis-ready PNGs are in `ms-thesis/images/` (`*_8b.png`, `*_1b.png`, `cross_model_comparison.png`, `cross_model_idk_behavior.png`).
