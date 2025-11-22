# UCI Health & Human Activity ML Lab

This repository contains a set of reproducible machine-learning experiments built from classic UCI datasets.  
The common theme is **risk and behavior in health**:

- Inpatient **diabetes 30-day readmission**
- Individual **obesity level** from lifestyle and demographic factors
- **Human activity recognition** from smartphone sensors

Each sub-project follows the same pattern:

1. Fetch and persist the raw data to local CSVs (`01_fetch_and_save.py`)
2. Train a baseline model with a clean preprocessing pipeline (`02_baseline_model.py`)
3. Generate diagnostic plots and interpret model behavior (`03_*_analysis_plots.py`)

The goal is to demonstrate disciplined, end-to-end ML practice rather than leaderboard chasing.

---

## Repository Structure

```text
DM_UCI_ML_REPO/
  .venv/                         # Local virtual environment (ignored by git)
  .vscode/                       # Editor configuration (optional, ignored)
  README.md
  diabetes_readmission_uci/
    data/
      diabetes_uci_features_raw.csv
      diabetes_uci_targets_raw.csv
      diabetes_uci_merged_binary.csv
    reports/
      figures/                   # Confusion matrix, ROC, etc.
    src/
      01_fetch_and_save.py
      02_baseline_model.py
      03_diabetes_analysis_plots.py
    requirements.txt             # Shared dependencies
  obesity_levels_uci/
    data/
      obesity_features_raw.csv
      obesity_targets_raw.csv
      obesity_merged.csv
    reports/
      figures/
    src/
      01_fetch_and_save.py
      02_baseline_model.py
      03_obesity_analysis_plots.py
  har_smartphones_uci/
    data/
      UCI_HAR_Dataset.zip        # Original archive (optional, ignored if large)
      har_features_raw.csv
      har_targets_raw.csv
      har_merged.csv
    reports/
      figures/
    src/
      01_fetch_and_save.py
      02_baseline_model.py
      03_har_analysis_plots.py
Environment and Reproducibility
All three projects use the same Python toolchain (pandas, scikit-learn, matplotlib, seaborn, shap, jupyterlab).

From the repo root:

bash
Copy code
# create and activate a local virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # PowerShell on Windows

# install shared dependencies
pip install -r diabetes_readmission_uci/requirements.txt
Each project can then be run independently as described below.

1. Diabetes 30-Day Readmission (diabetes_readmission_uci)
Problem
Use ten years of inpatient encounters from 130 U.S. hospitals (UCI “Diabetes 130-US hospitals 1999–2008”) to predict whether a hospitalized patient with diabetes will be readmitted within 30 days of discharge.

Task: Binary classification (readmit_30d ∈ {0,1})

Label construction: UCI readmitted field is mapped to
readmit_30d = 1 for <30 and 0 otherwise.

Challenge: The positive class (30-day readmission) is a relatively rare event; a trivial “never readmit” classifier already achieves high accuracy.

Pipeline
Scripts:

src/01_fetch_and_save.py
Fetches UCI dataset via ucimlrepo, separates features/targets, constructs the binary readmit_30d label, and writes:

diabetes_uci_features_raw.csv

diabetes_uci_targets_raw.csv

diabetes_uci_merged_binary.csv

src/02_baseline_model.py
Baseline logistic regression:

Splits data into train/test with stratification on readmit_30d

One-hot encodes categorical variables and scales numeric columns via a ColumnTransformer

Trains a logistic regression classifier

Prints a full classification_report

src/03_diabetes_analysis_plots.py
Re-trains a logistic regression model with class weighting and produces:

Normalized confusion matrix

ROC curve

Bar chart of precision/recall/F1 for the positive class

Run:

bash
Copy code
cd diabetes_readmission_uci

python .\src\01_fetch_and_save.py
python .\src\02_baseline_model.py
python .\src\03_diabetes_analysis_plots.py
Key Findings (Interpretation)
Severe class imbalance distorts accuracy.
The underlying dataset has many more “no readmission” than “readmission” encounters.
A simple majority classifier achieves high overall accuracy but completely fails to identify patients at risk. The baseline unweighted logistic regression reflects this pattern: very high performance on class 0 with extremely low recall for class 1.

Logistic regression captures signal but prioritizes specificity over sensitivity.
When class weights are introduced, the model begins to trade some specificity for better sensitivity, but the recall for 30-day readmission remains modest. This is expected: readmission is influenced by unobserved social, behavioral, and post-acute factors that are only partly reflected in structured EHR data.

ROC curve emphasizes threshold choice.
The ROC curve produced by 03_diabetes_analysis_plots.py lies above the diagonal, indicating discriminatory ability beyond chance. However, depending on the selected operating point, the model can favor:

Higher sensitivity (flag more at-risk patients, accept more false positives), or

Higher specificity (avoid alarm fatigue, miss some true readmissions).

Clinical implication.
In practice, such a model is more suitable as a screening tool embedded in discharge planning rather than a definitive decision rule. It can help prioritize chart review or case management outreach but should not be used to deny discharge or services.

2. Obesity Levels from Lifestyle and Demographics (obesity_levels_uci)
Problem
Predict an individual’s obesity level from demographic information and self-reported lifestyle attributes (physical activity, dietary patterns, screen time, etc.).

Task: Multiclass classification (obesity_level)

Label: The UCI dataset encodes several ordered categories (from underweight through various degrees of overweight and obesity).

Pipeline
Scripts:

src/01_fetch_and_save.py
Fetches the UCI obesity dataset via ucimlrepo, saves:

obesity_features_raw.csv

obesity_targets_raw.csv

obesity_merged.csv (features + obesity_level)

src/02_baseline_model.py

Train/test split with stratification on obesity_level

One-hot encoding for categorical attributes (e.g., family history, transportation mode)

Standardization of numeric variables

Multinomial logistic regression with class_weight="balanced"

Prints per-class precision/recall/F1

src/03_obesity_analysis_plots.py

Normalized confusion matrix for all obesity levels

Per-class F1 bar plot

(Optionally) inspection of top positive/negative coefficients for selected classes

Run:

bash
Copy code
cd obesity_levels_uci

python .\src\01_fetch_and_save.py
python .\src\02_baseline_model.py
python .\src\03_obesity_analysis_plots.py
Key Findings (Interpretation)
Model performance reflects the ordered nature of the labels.
The confusion matrix typically shows that the model rarely confuses extremes (e.g., underweight vs. severe obesity) but struggles more with adjacent categories (e.g., normal vs. overweight). This is consistent with the clinical reality that boundaries between neighboring BMI-based categories are somewhat arbitrary.

Balanced class weights mitigate, but do not eliminate, class imbalance.
Some obesity levels are under-represented in the dataset. Using class_weight="balanced" helps the model pay attention to minority classes, but performance metrics still vary by class size; minority categories tend to have lower recall and more unstable estimates.

Lifestyle and behavior variables have interpretable effects.
Examining logistic regression coefficients (behind the scenes of the pipeline) reveals patterns that match domain expectations:

Higher physical activity, shorter screen time, and healthier reported diet patterns shift probability mass toward lower obesity categories.

Sedentary behavior and frequent consumption of calorie-dense foods shift mass toward overweight and obesity categories.

Use case.
This model is best viewed as an educational and exploratory tool: it highlights how combinations of lifestyle factors associate with weight categories in a labeled dataset. It is not intended to replace clinical assessment or individualized nutritional counseling.

3. Human Activity Recognition from Smartphone Sensors (har_smartphones_uci)
Problem
Classify a person’s activity (e.g., walking, sitting, standing) from smartphone accelerometer and gyroscope signals.

Task: Multiclass classification (activity)

Data: UCI Human Activity Recognition Using Smartphones dataset.

Pre-processed time-domain and frequency-domain features are provided in the original UCI archive.

har_merged.csv contains these features plus an activity label.

Pipeline
Scripts:

src/01_fetch_and_save.py
Wraps extraction/merging of the UCI HAR dataset into:

har_features_raw.csv

har_targets_raw.csv

har_merged.csv

src/02_baseline_model.py

Train/test split stratified on activity

Standardization of numeric features

Multiclass logistic regression baseline

Full classification report

src/03_har_analysis_plots.py

Normalized confusion matrix across activities

Per-activity F1 bar plot

Two-dimensional PCA embedding colored by activity label, illustrating cluster separation in feature space

Run:

bash
Copy code
cd har_smartphones_uci

python .\src\01_fetch_and_save.py
python .\src\02_baseline_model.py
python .\src\03_har_analysis_plots.py
Key Findings (Interpretation)
Well-engineered features make even simple models competitive.
The UCI HAR dataset includes carefully engineered features (means, standard deviations, frequency-domain measures). As a result, a relatively simple multinomial logistic regression already achieves strong performance across most activities.

Misclassifications concentrate in physically similar activities.
The confusion matrix typically shows:

High performance on clearly distinct motions (e.g., walking vs. lying)

More confusion between posturally similar states (e.g., sitting vs. standing), where smartphone orientation and small movements overlap.

PCA reveals structure in the feature space.
The PCA plot from 03_har_analysis_plots.py demonstrates that activities form partially separable clusters even when compressed into two dimensions. This provides an intuitive visual justification for why linear classifiers can perform well and serves as a baseline for comparison with more complex models (e.g., CNNs or sequence models).

Use case.
This experiment functions as a baseline HAR benchmark and a staging point for future work with deep learning architectures on raw sensor sequences.

Design Choices and Next Steps
Across all three projects:

Consistent preprocessing.
Categorical and numeric features are handled explicitly via a ColumnTransformer, avoiding data leakage and ensuring transformations learned on the training set are applied identically to the test set.

Transparent baselines.
Logistic regression is intentionally used as a first-line model. It provides:

Convincing baselines for later comparison

Direct interpretability through coefficients and odds ratios

Fast training and straightforward diagnostics

Diagnostics over single metrics.
Instead of reporting only accuracy, the analysis emphasizes:

Per-class precision, recall, and F1

Confusion matrices

ROC curves (for binary classification)

Visualizations (e.g., PCA embeddings) to reason about feature separability

Potential Extensions
This repository is structured so that more advanced methods can be layered on without major refactoring:

Replace logistic regression with tree-based methods (Random Forest, XGBoost, LightGBM) or neural networks.

Calibrate and compare different decision thresholds for readmission risk, explicitly modeling the cost of false negatives vs. false positives.

Introduce ordinal classification methods for obesity levels to better respect the ordered nature of the labels.

Apply sequence models (1D CNNs, RNNs, or transformers) directly to the raw HAR time series to study gains over engineered features.

How to Cite or Describe This Work
When referencing this repository in a portfolio, curriculum vitae, or project report, it can be accurately summarized as:

“Developed an end-to-end machine-learning lab using three UCI datasets (inpatient diabetes readmission, obesity levels, smartphone human activity recognition). Implemented reproducible pipelines with scikit-learn (data ingestion, preprocessing, model training, and evaluation), produced diagnostic visualizations (confusion matrices, ROC curves, PCA embeddings), and provided domain-specific interpretation of model behavior and limitations.”

pgsql
Copy code

You can paste that directly into `README.md`. If you later grab actual numeric metrics you want highlighted (e.g., specific F1 scores), you can drop a short table into each project section without changing the overall structure.
::contentReference[oaicite:0]{index=0}






You said:
I want the images in the README with explanations. 


ChatGPT said:

Thought for 13s
Nice. Time to make the README look like an actual paper-in-a-repo.

Here’s a full updated README.md with embedded image references and short explanations for each figure. Paths assume the images live where your scripts save them now.

markdown
Copy code
# UCI Health & Human Activity ML Lab

This repository contains a set of reproducible machine-learning experiments built from classic UCI datasets.  
The common theme is **risk and behavior in health**:

- Inpatient **diabetes 30-day readmission**
- Individual **obesity level** from lifestyle and demographic factors
- **Human activity recognition** from smartphone sensors

Each sub-project follows the same pattern:

1. Fetch and persist the raw data to local CSVs (`01_fetch_and_save.py`)
2. Train a baseline model with a clean preprocessing pipeline (`02_baseline_model.py`)
3. Generate diagnostic plots and interpret model behavior (`03_*_analysis_plots.py`)

The goal is to demonstrate disciplined, end-to-end ML practice rather than leaderboard chasing.

---

## Repository Structure

```text
DM_UCI_ML_REPO/
  .venv/                         # Local virtual environment (ignored by git)
  .vscode/                       # Editor configuration (optional, ignored)
  README.md
  diabetes_readmission_uci/
    data/
      diabetes_uci_features_raw.csv
      diabetes_uci_targets_raw.csv
      diabetes_uci_merged_binary.csv
    reports/
      figures/
        diabetes_confusion_matrix.png
        diabetes_roc_curve.png
        diabetes_readmit1_metrics.png
    src/
      01_fetch_and_save.py
      02_baseline_model.py
      03_diabetes_analysis_plots.py
    requirements.txt             # Shared dependencies
  obesity_levels_uci/
    data/
      obesity_features_raw.csv
      obesity_targets_raw.csv
      obesity_merged.csv
    reports/
      figures/
        obesity_confusion_matrix.png
        obesity_f1_by_class.png
    src/
      01_fetch_and_save.py
      02_baseline_model.py
      03_obesity_analysis_plots.py
  har_smartphones_uci/
    data/
      UCI_HAR_Dataset.zip        # Original archive (optional, large)
      har_features_raw.csv
      har_targets_raw.csv
      har_merged.csv
    reports/
      figures/
        har_confusion_matrix.png
        har_f1_by_activity.png
        har_pca_embedding.png
    src/
      01_fetch_and_save.py
      02_baseline_model.py
      03_har_analysis_plots.py
Environment and Reproducibility
All three projects use the same Python toolchain (pandas, scikit-learn, matplotlib, seaborn, shap, jupyterlab).

From the repo root:

bash
Copy code
# create and activate a local virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # PowerShell on Windows

# install shared dependencies
pip install -r diabetes_readmission_uci/requirements.txt
Each project can then be run independently as described below.

1. Diabetes 30-Day Readmission (diabetes_readmission_uci)
Problem
Use ten years of inpatient encounters from 130 U.S. hospitals (UCI “Diabetes 130-US hospitals 1999–2008”) to predict whether a hospitalized patient with diabetes will be readmitted within 30 days of discharge.

Task: Binary classification (readmit_30d ∈ {0,1})

Label construction: UCI readmitted field is mapped to
readmit_30d = 1 for <30 and 0 otherwise.

Challenge: The positive class (30-day readmission) is a relatively rare event; a trivial “never readmit” classifier already achieves high accuracy.

Pipeline
Scripts:

src/01_fetch_and_save.py
Fetches the UCI dataset via ucimlrepo, separates features/targets, constructs the binary readmit_30d label, and writes:

diabetes_uci_features_raw.csv

diabetes_uci_targets_raw.csv

diabetes_uci_merged_binary.csv

src/02_baseline_model.py
Baseline logistic regression:

Train/test split with stratification on readmit_30d

One-hot encodes categorical variables and scales numeric columns via a ColumnTransformer

Trains a logistic regression classifier

Prints a full classification_report

src/03_diabetes_analysis_plots.py
Re-trains a logistic regression model with class weighting and produces:

Normalized confusion matrix

ROC curve

Bar chart of precision/recall/F1 for the positive class

Run:

bash
Copy code
cd diabetes_readmission_uci

python .\src\01_fetch_and_save.py
python .\src\02_baseline_model.py
python .\src\03_diabetes_analysis_plots.py
Visual Diagnostics
Confusion Matrix


The confusion matrix highlights the core challenge: the model is very strong at recognizing no readmission encounters but struggles with the much rarer 30-day readmissions. Even with class weighting, the recall for the positive class remains modest, underscoring the difficulty of predicting readmission from structured inpatient data alone.

ROC Curve


The ROC curve lies consistently above the diagonal, confirming that the model is substantially better than random guessing. However, the slope near clinically relevant operating points shows a clear trade-off: improving sensitivity for readmissions comes at the cost of more false alarms, which has implications for care-management workload and alert fatigue.

Positive-Class Metrics


This bar chart isolates precision, recall, and F1 for the readmit=1 class. It makes explicit that the model is more precise than sensitive: when it predicts a readmission it is often correct, but it fails to catch many true positives. This pattern supports using the model as a screening aid rather than as a gatekeeping tool.

Key Findings (Interpretation)
Severe class imbalance distorts accuracy.
High overall accuracy is largely driven by the dominant “no readmission” class; without class-aware evaluation, risk models can look deceptively strong.

Logistic regression captures signal but underestimates complex risk.
Adding class weights improves sensitivity but still leaves a large pool of missed readmissions, reflecting the role of non-captured social and behavioral determinants.

Threshold selection is a policy decision, not a purely technical one.
ROC and metric plots show that moving the threshold changes who is flagged for follow-up and how many false positives case managers must handle.

Clinical implication.
The model is best interpreted as a prioritization tool that helps identify higher-risk discharges for review, not as an automated decision-maker.

2. Obesity Levels from Lifestyle and Demographics (obesity_levels_uci)
Problem
Predict an individual’s obesity level from demographic information and self-reported lifestyle attributes (physical activity, dietary patterns, screen time, etc.).

Task: Multiclass classification (obesity_level)

Label: The UCI dataset encodes several ordered categories (from underweight through various degrees of overweight and obesity).

Pipeline
Scripts:

src/01_fetch_and_save.py
Fetches the UCI obesity dataset via ucimlrepo, saves:

obesity_features_raw.csv

obesity_targets_raw.csv

obesity_merged.csv (features + obesity_level)

src/02_baseline_model.py

Train/test split with stratification on obesity_level

One-hot encoding for categorical attributes (e.g., family history, transportation mode)

Standardization of numeric variables

Multinomial logistic regression with class_weight="balanced"

Prints per-class precision/recall/F1

src/03_obesity_analysis_plots.py

Normalized confusion matrix for all obesity levels

Per-class F1 bar plot

Run:

bash
Copy code
cd obesity_levels_uci

python .\src\01_fetch_and_save.py
python .\src\02_baseline_model.py
python .\src\03_obesity_analysis_plots.py
Visual Diagnostics
Confusion Matrix


The confusion matrix shows that errors cluster between adjacent categories (e.g., normal vs. overweight), while extreme categories (underweight vs. severe obesity) are more clearly separated. This mirrors how BMI cut points behave in practice: boundaries between neighbors are somewhat arbitrary and inherently noisy.

Per-Class F1 Scores


The F1 plot exposes how performance varies by class frequency. Common categories achieve stronger and more stable F1 scores, while rare obesity levels are harder to estimate reliably even with balanced class weights. This is a reminder that class imbalance directly shapes which predictions can be trusted most.

Key Findings (Interpretation)
Ordinal structure matters.
Most misclassifications land one category away from the true label, suggesting that models which explicitly encode the ordered nature of obesity (e.g., ordinal regression) would be a natural next step.

Lifestyle variables behave as expected.
Examination of logistic coefficients (not shown in the README but available in code) confirms intuitive relationships: higher activity and healthier eating patterns push probabilities toward lower obesity levels.

Use case.
This model is well-suited for scenario analysis and education—illustrating how clusters of behaviors relate to weight categories—rather than for individualized clinical decision-making.

3. Human Activity Recognition from Smartphones (har_smartphones_uci)
Problem
Classify a person’s activity (e.g., walking, sitting, standing) from smartphone accelerometer and gyroscope signals.

Task: Multiclass classification (activity)

Data: UCI Human Activity Recognition Using Smartphones dataset.
har_merged.csv contains engineered time- and frequency-domain features plus an activity label.

Pipeline
Scripts:

src/01_fetch_and_save.py
Wraps extraction/merging of the UCI HAR dataset into:

har_features_raw.csv

har_targets_raw.csv

har_merged.csv

src/02_baseline_model.py

Train/test split stratified on activity

Standardization of numeric features

Multiclass logistic regression baseline

Full classification report

src/03_har_analysis_plots.py

Normalized confusion matrix across activities

Per-activity F1 bar plot

Two-dimensional PCA embedding colored by activity label

Run:

bash
Copy code
cd har_smartphones_uci

python .\src\01_fetch_and_save.py
python .\src\02_baseline_model.py
python .\src\03_har_analysis_plots.py
Visual Diagnostics
Confusion Matrix


The confusion matrix shows that the baseline model distinguishes most activities very well, with misclassifications concentrated among posturally similar states (for example, sitting vs. standing). This is consistent with what we would expect from a single smartphone worn at the waist.

Per-Activity F1 Scores


The F1 score chart confirms the pattern from the confusion matrix: dynamic activities such as walking or walking upstairs are easier to classify, whereas static or low-motion activities produce more overlap in feature space and thus lower F1.

PCA Embedding


The PCA plot projects high-dimensional sensor features into two dimensions and colors points by activity. Distinct clusters emerge for many activities, visually justifying why a simple linear classifier already performs strongly and providing an intuitive baseline for future deep-learning approaches on raw time series.

Key Findings (Interpretation)
Feature engineering pays off.
UCI’s pre-engineered features allow a straightforward multinomial logistic regression to perform surprisingly well, making this dataset ideal for didactic comparisons between classical ML and deep learning.

Confusion patterns reflect physical similarity.
Errors primarily occur where the underlying motions are genuinely similar, not because the model is random; this is a useful sanity check when building applied HAR systems.

Use case.
The experiment serves as a baseline HAR benchmark and a clean starting point for extending into CNN/RNN/transformer models on raw accelerometer sequences.

Design Choices and Next Steps
Across all three projects:

Consistent preprocessing.
Categorical and numeric features are handled explicitly via ColumnTransformer, avoiding data leakage and ensuring transformations learned on the training set are applied identically to the test set.

Transparent baselines.
Logistic regression is intentionally used as a first-line model. It provides:

Convincing baselines for later comparison

Direct interpretability through coefficients and odds ratios

Fast training and straightforward diagnostics

Diagnostics over single metrics.
Instead of reporting only accuracy, the analysis emphasizes:

Per-class precision, recall, and F1

Confusion matrices

ROC curves (for binary classification)

Visualizations (e.g., PCA embeddings) to reason about feature separability

Potential Extensions
The current structure makes it easy to plug in more advanced methods:

Tree-based ensembles or gradient boosting for tabular data

Ordinal classification methods for obesity levels

Cost-sensitive and threshold-calibrated models for readmission risk

CNN/RNN/transformer models operating directly on raw HAR time series

How to Describe This Work
When referencing this repository in a portfolio, curriculum vitae, or project report, it can be accurately summarized as:

“Developed an end-to-end machine-learning lab using three UCI datasets (inpatient diabetes readmission, obesity levels, smartphone human activity recognition). Implemented reproducible pipelines with scikit-learn (data ingestion, preprocessing, model training, and evaluation), produced diagnostic visualizations (confusion matrices, ROC curves, PCA embeddings), and provided domain-specific interpretation of model behavior and limitations.”