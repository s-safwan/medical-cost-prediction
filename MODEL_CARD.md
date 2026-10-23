# Model card

## Purpose
Educational prediction of annual medical charges from six tabular features. No insurance premium or coverage recommendation is computed.

## Features
Age, BMI, number of children, gender (the source field label), smoking status, and region. Demographic variables can encode sensitive associations. Their inclusion is for studying the supplied data, not an endorsement of use in underwriting.

## Evaluation
Exact duplicates are removed before an 80/20 seeded split stratified by smoking status. Five shuffled folds on the 1,069 training records compare a median baseline, Ridge regression, random forest and gradient boosting by MAE. Imputation, scaling and encoding are fitted inside each fold. Lowest mean CV MAE determines the model. The 268 test records are then used for final evaluation. Baseline and Ridge use those same test records for comparisons.

CV fold standard deviation is not a confidence interval. Permutation importance is a post-selection diagnostic, not a causal effect. Small subgroup counts and differing target distributions prevent a fairness conclusion from subgroup MAE alone.

## Limitations
Source, license, population, units and collection period are unconfirmed. No geographic, temporal, prospective or clinical validation was performed. Target skew leaves large errors on some records; no calibrated individual prediction intervals are provided. Further model development after seeing these test results would require a fresh final evaluation set. No regulatory compliance, deployment success, or business impact is demonstrated.
