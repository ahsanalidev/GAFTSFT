# Result Analysis

## Strongest findings from the current JSON summaries

- GAFT has the strongest Forget Quality (0.5812 $\pm$ 0.0000) across 1 runs.
- IDK has the lowest mean MIA score (0.3840 $\pm$ 0.0000), improving over GAFT by 0.0558 and over PGA by 0.2232.
- GAFT keeps the best overall utility average across non-forget splits (0.7909 $\pm$ 0.0000), with especially strong real-author accuracy (0.9300 $\pm$ 0.0000).
- PGA has the worst privacy profile: highest mean MIA (0.6072 $\pm$ 0.0000) and weakest Forget Quality (0.0541 $\pm$ 0.0000).
- IDK is still the most suppression-oriented method, lowering forget accuracy to 0.5175 $\pm$ 0.0000 while also showing the lowest retain accuracy (0.5425 $\pm$ 0.0000).

## Suggested paper additions

- Add a cross-split utility table that includes retain, real-author, and world-fact metrics, not just forget-set values.
- Add a Forget Quality vs. MIA scatter plot to show that GAFT and IDK dominate PGA in different ways.
- Add an MIA-threshold line chart to show that IDK consistently stays below the other methods across all min-k thresholds.
- Add a short qualitative failure analysis using the generated forget answers: several responses still hallucinate substitute biographies instead of refusing cleanly.

## Manuscript check

- The paper should now report aggregated numbers, not a single run. The current PGA Forget Quality aggregate is 0.0541 $\pm$ 0.0000.
