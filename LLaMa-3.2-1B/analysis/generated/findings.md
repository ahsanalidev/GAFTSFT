# Result Analysis

## Strongest findings from the current JSON summaries

- GAFT has the strongest Forget Quality (0.5610 $\pm$ 0.0381) across 3 runs.
- IDK has the lowest mean MIA score (0.4251 $\pm$ 0.0076), improving over GAFT by 0.0265 and over PGA by 0.2169.
- GAFT keeps the best overall utility average across non-forget splits (0.7112 $\pm$ 0.0120), with especially strong real-author accuracy (0.8206 $\pm$ 0.0079).
- PGA has the worst privacy profile: highest mean MIA (0.6420 $\pm$ 0.0167) and weakest Forget Quality (0.0408 $\pm$ 0.0023).
- IDK is still the most suppression-oriented method, lowering forget accuracy to 0.5369 $\pm$ 0.0058 while also showing the lowest retain accuracy (0.5122 $\pm$ 0.0017).

## Suggested paper additions

- Add a cross-split utility table that includes retain, real-author, and world-fact metrics, not just forget-set values.
- Add a Forget Quality vs. MIA scatter plot to show that GAFT and IDK dominate PGA in different ways.
- Add an MIA-threshold line chart to show that IDK consistently stays below the other methods across all min-k thresholds.
- Add a short qualitative failure analysis using the generated forget answers: several responses still hallucinate substitute biographies instead of refusing cleanly.

## Manuscript check

- The paper should now report aggregated numbers, not a single run. The current PGA Forget Quality aggregate is 0.0408 $\pm$ 0.0023.
