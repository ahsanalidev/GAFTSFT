# Result Analysis

## Strongest findings from the current JSON summaries

- GAFT has the strongest Forget Quality (0.5427 $\pm$ 0.0334) across 3 runs.
- IDK has the lowest mean MIA score (0.4171 $\pm$ 0.0070), improving over GAFT by 0.0247 and over PGA by 0.2048.
- GAFT keeps the best overall utility average across non-forget splits (0.7912 $\pm$ 0.0107), with especially strong real-author accuracy (0.9400 $\pm$ 0.0100).
- PGA has the worst privacy profile: highest mean MIA (0.6219 $\pm$ 0.0130) and weakest Forget Quality (0.0392 $\pm$ 0.0046).
- IDK is still the most suppression-oriented method, lowering forget accuracy to 0.5617 $\pm$ 0.0063 while also showing the lowest retain accuracy (0.5450 $\pm$ 0.0050).

## Suggested paper additions

- Add a cross-split utility table that includes retain, real-author, and world-fact metrics, not just forget-set values.
- Add a Forget Quality vs. MIA scatter plot to show that GAFT and IDK dominate PGA in different ways.
- Add an MIA-threshold line chart to show that IDK consistently stays below the other methods across all min-k thresholds.
- Add a short qualitative failure analysis using the generated forget answers: several responses still hallucinate substitute biographies instead of refusing cleanly.

## Manuscript check

- The paper should now report aggregated numbers, not a single run. The current PGA Forget Quality aggregate is 0.0392 $\pm$ 0.0046.
