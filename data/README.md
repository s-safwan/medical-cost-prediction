# Dataset access and provenance

Training expects `data/medical.xlsx` with columns `age`, `gender`, `bmi`, `children`, `smoker`, `region`, `charges`.

Raw workbook: 1,339 rows; one invalid-target trailing row and one exact duplicate removed; 1,337 usable records. Numeric feature missing values are imputed inside training folds. Content SHA-256 is recorded in `reports/metrics.json`.
