# Student Performance Analytics — Notebook Outline

1. Load `../data/Expanded_data_with_more_features.csv`
2. Inspect shape, dtypes, descriptive statistics and missing values.
3. Remove index columns and standardize blank strings as missing.
4. Convert `NrSiblings` and score columns to numeric.
5. Impute categorical values with `Unknown` and numeric values with median.
6. Create `AverageScore` and `PerformanceBand`.
7. Analyze score distributions and grouped means for `ParentEduc`, `TestPrep`, `LunchType`, `WklyStudyHours`, `Gender`, etc.
8. Build a correlation matrix.
9. Train a Ridge regression pipeline with one-hot encoding and scaling.
10. Evaluate with MAE and R².
11. Compare actual vs predicted scores.
12. Summarize business/education implications without claiming causation.
