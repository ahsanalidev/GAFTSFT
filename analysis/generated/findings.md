# Result Analysis

## Strongest findings from the current JSON summaries

- GAFT has the strongest Forget Quality (0.5812), far ahead of IDK (0.1111) and PGA (0.0541).
- IDK has the lowest mean MIA score (0.3840), improving over GAFT by 0.0558 and over PGA by 0.2232.
- GAFT keeps the best overall utility average across non-forget splits (0.7909), with especially strong real-author accuracy (0.9300).
- PGA has the worst privacy profile: highest mean MIA (0.6072) and weakest Forget Quality (0.0541).
- IDK is best described as a suppression-oriented method, because it lowers forget accuracy to 0.5175 and forget ROUGE to 0.3818, but it also has the lowest retain accuracy (0.5425) and world-fact accuracy (0.8547).

## Suggested paper additions

- Add a cross-split utility table that includes retain, real-author, and world-fact metrics, not just forget-set values.
- Add a Forget Quality vs. MIA scatter plot to show that GAFT and IDK dominate PGA in different ways.
- Add an MIA-threshold line chart to show that IDK consistently stays below the other methods across all min-k thresholds.
- Add a short qualitative failure analysis using the generated forget answers: several responses still hallucinate substitute biographies instead of refusing cleanly.

## Manuscript check

- The current paper table reports PGA Forget Quality as 0.0299, but the latest `results/PureGradientAscent/important.json` stores 0.0541. This looks like a stale number and should be reconciled before submission.
