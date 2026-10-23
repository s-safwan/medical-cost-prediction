# Upload and application guide

## GitHub (browser upload)
1. Create a repository at https://github.com/new under s-safwan.
2. Repository name: medical-cost-prediction
3. Description: Leakage-safe medical cost regression with reproducible model comparison, held-out evaluation and error analysis.
4. Choose Public. Do not add an initial README, license or .gitignore; these project files include the README and .gitignore. Create repository.
5. Extract the delivered ZIP. Open the medical-cost-prediction folder.
6. On GitHub, select "uploading an existing file" (or Add file > Upload files). Drag the CONTENTS of that folder, including reports, tests and data folders. Do not upload the ZIP itself or the enclosing folder. Include .gitignore if your file browser hides it.
7. Commit message: Add reproducible medical cost prediction study
8. Verify README renders the chart and reports/metrics.json opens.
9. In About, add topics: python, machine-learning, scikit-learn, regression, gradient-boosting, data-analysis
10. On your GitHub profile, use Customize your pins to pin this repository alongside FlyRank and your strongest SQL/BI project.

No dataset or serialized model is in the upload package. To reproduce locally, copy medical.xlsx from your ORIGINAL semester ZIP into data/medical.xlsx, install requirements, and run python train.py. Confirm its original source and license before distributing the workbook. The local demo requires requirements-demo.txt and a runtime check before you advertise it as a working app.

## Internship profile: project fields
Title: Medical Cost Prediction with Leakage-Safe Model Evaluation
Role: Individual semester project; portfolio revision
Project URL (after upload): https://github.com/s-safwan/medical-cost-prediction
Technologies: Python, pandas, scikit-learn, gradient boosting, random forest, matplotlib

Short description:
Developed a reproducible regression workflow to estimate medical charges from six tabular features. Validated 1,337 records, compared four models using training-only five-fold cross-validation, and achieved held-out R² of 0.925 and MAE of 1,404 on 268 test records. Reduced MAE by 61% versus Ridge on the same split; documented subgroup errors, feature importance and model limitations.

If there is a character limit, use:
Built a leakage-safe medical cost regression pipeline; selected gradient boosting using five-fold CV and achieved held-out R² 0.925, reducing MAE 61% vs Ridge on the same split. Added reproducible reports and error analysis.

Dates: Enter your actual semester start/end dates; do not invent them.
Status: Completed educational study; optional local demo, not deployed.
Do not list as published research or an insurance pricing product.

## Interview preparation
- Why MAE? It measures absolute charge error in understandable units; RMSE also shows sensitivity to large errors.
- Why nonlinear models? They can capture interactions, such as smoking status and BMI, without manually specifying them. Importance is association, not causation.
- How was leakage reduced? Duplicates removed before the split; preprocessing fitted inside folds; candidate choice used only training CV.
- What is still weak? Small data, unknown source/license, one test split, large outliers, no external validation, no individual uncertainty intervals.
- Why remove premium tiers? Their percentage rates were unsupported. Predicting observed charges does not establish an insurance premium.
- Which numbers can I compare? The revised model, median and Ridge share the new test split. The old notebook used a different split.

## Next work item
FlyRank repository located: https://github.com/s-safwan/ml-internship-flyrank
It has not been edited in this pass. Supply the research paper or its exact repository path, plus the internship profile URL or field labels, so its research status and claims can be checked before writing project/research entries.
