# Dataset access and provenance

Training expects `data/medical.xlsx` with columns `age`, `gender`, `bmi`, `children`, `smoker`, `region`, `charges`.

The workbook was supplied with the author's semester project. Its upstream source, license, units, population, collection period, and whether it is synthetic have not been independently established. The old notebook called it synthetic; that statement is unverified. Do not present it as newly collected clinical data.

The private review bundle retains the supplied workbook for reproduction. The public upload excludes the workbook and original notebook pending source/license confirmation. Add the legitimate original download link and license here once verified. No source is invented.

Raw workbook: 1,339 rows; one invalid-target trailing row and one exact duplicate removed; 1,337 usable records. Numeric feature missing values are imputed inside training folds. Content SHA-256 is recorded in `reports/metrics.json`.
