# Recorded results and provenance

`recorded_metrics.csv` is transcribed programmatically from saved execution text of the final experiment. It contains the last three-model comparison, retaining the printed four-decimal precision. Values written only in comments for earlier trial settings are not included.

`recorded_regression.json` records the ridge model's score on the same data used to fit it, using the IQR-clipped continuous target. It is an in-sample R2, not a held-out regression result.

`figures/model_comparison.png` can be regenerated using `python engagement_prediction.py --plot-recorded`. `figures/gradient_boosting_feature_importance.png` is an existing saved output figure, not a newly estimated importance plot. Its values are impurity-based importances and do not establish causal relationships.

The historical classification protocol uses a stratified 80/20 split with random state 42 after fitting preprocessing on all rows. The regression target is also clipped during preprocessing. Model settings were compared using the same test split, which therefore was not an untouched final holdout. These limitations restrict generalization claims.

New runs write to `runs/latest/` by default and do not overwrite the recorded measurements. The original environment versions are unavailable, so matching the printed metrics exactly is not guaranteed.

## Standalone validation run

`validation_metrics.csv` contains the new end-to-end run of the same retained model configurations on the supplied input. `validation_summary.json` records its Python/package versions, sample counts, feature count, class proportion, target changes, and in-sample ridge score. No raw rows, input paths, or model weights are included.

Logistic Regression matches the historical four-decimal scores. Tree-model scores differ slightly; no claim of exact reproduction is made. This run retains the same preprocessing leakage and target clipping and must not be interpreted as a corrected evaluation protocol. The historical chart and feature-importance figure continue to describe the historical run.
