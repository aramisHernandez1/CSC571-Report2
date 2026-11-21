# Predicting Negative Steam Reviews from Review Text

CSC 576 · Data Science · Fall 2026 · Report 2
Author: Aramis Hernandez

## What this project does

Can I predict whether a Steam review recommends a game (positive, +1) or not (negative, -1) from the written text of the review alone? This is a binary classification task. About 82% of reviews are positive, so a model that always guesses "positive" scores about 82% accuracy. Every model is therefore compared against that baseline and judged mainly on precision, recall and F1 for the **negative** class.

Models compared on one held-out test set:

1. Baseline: `DummyClassifier(strategy="most_frequent")`
2. Logistic regression (default class weights)
3. Logistic regression (`class_weight="balanced"`), the preferred model
4. Random forest (100 trees, `min_samples_leaf=2`, `class_weight="balanced_subsample"`)

Text is converted to numbers with TF-IDF (10,000 words, `min_df=5`, single words only).

## Project layout

```
.
├── README.md
├── requirements.txt
├── data/        cleaned sample (if small enough) or empty, see "Data" below
├── src/
│   ├── cleanData.ipynb   step 1: download, filter, sample, de-duplicate
│   └── Main.ipynb        step 2: split, TF-IDF, train models, results, figures
├── results/     results table(s) saved by the code
└── figures/     every figure used in the report, saved by the code
```

## Data

- Source: Kaggle, "Steam Reviews" (6.4M reviews, 2.16 GB): https://www.kaggle.com/datasets/andrewmvd/steam-reviews
- Original dataset citation: Antoni Sobkowicz (2017), Steam Review Dataset, Zenodo, https://doi.org/10.5281/zenodo.1000885. License: CC BY-NC 4.0.
- The raw file is **not included** because it is far over 25 MB. The code downloads it automatically with `kagglehub` (no Kaggle account or API key should be needed for this public dataset; the first run downloads about 2.16 GB and may take several minutes).
- One row = one Steam review. Columns: `app_id`, `app_name`, `review_text`, `review_score` (+1/-1), `review_votes`.
- Only `review_text` (feature) and `review_score` (target) are used for modeling.

## How to run

Tested with Python 3.14 and scikit-learn 1.9.1. Other recent Python 3 versions should work.

```bash
# 1. Unzip the project and open a terminal in the project folder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Start Jupyter
jupyter notebook
```

Then run the notebooks **in this order**, using *Kernel -> Restart & Run All* in each:

1. `src/cleanData.ipynb`
   Downloads the dataset, removes missing, empty and "Early Access Review" placeholder reviews, randomly samples 200,000 reviews (`seed = 42`), removes duplicate review texts (case-insensitive), and leaves 193,742 reviews.
2. `src/Main.ipynb`
   Makes the 80/20 stratified split (`random_state=42`), fits TF-IDF on the training set only, trains the baseline and three models, scores them on the test set, and saves the results table and figures.

All random choices use seed 42, so the numbers should be identical on every run with the same package versions.

Note: if you run from a different folder, check that `results/` and `figures/` end up at the project root, not inside `src/`.

## Outputs

| File | What it is |
|---|---|
| `results/results_table.csv` | Accuracy and negative-class precision, recall, F1 (plus ROC AUC) for every model on the test set |
| `results/validation_results.csv` | Model settings compared on a validation slice of the training data (if present) |
| `figures/fig1b_negative_class_scores.png` | Report Figure 1: negative-class precision/recall/F1 per model |
| `figures/fig1_confusion_matrices.png` | Report Figure 2: confusion matrices for all four models |
| `figures/fig2_feature_reliance.png` | Report Figure 3: most influential words (logistic regression coefficients and random forest importances) |

## Results (test set, 38,749 reviews)

| Model | Accuracy | Negative precision | Negative recall | Negative F1 |
|---|---|---|---|---|
| Baseline (always positive) | 82.2% | 0.000 | 0.000 | 0.000 |
| Logistic regression | 89.7% | 0.787 | 0.575 | 0.664 |
| Logistic regression (balanced) | 86.0% | 0.572 | 0.842 | 0.681 |
| Random forest | 87.3% | 0.694 | 0.508 | 0.587 |

Every model beats the baseline on negative-class recall and F1. Balanced logistic regression finds the most negative reviews (84.2% recall) at the cost of more false alarms. The random forest was the weakest model on negative reviews.

## Design choices that avoid leakage

- The train/test split is made before any fitting, and every reported score is on the test set.
- The TF-IDF vectorizer is fit on training text only, then applied to the test text.
- `review_votes` (only known after others read the review) and `app_id` / `app_name` (the model could learn which games are liked instead of reading text) are not used as features.
- There is no date column, so a random split is used. A game can appear in both train and test sets, which is listed as a limitation in the report.

## Limitations

TF-IDF on single words cannot capture negation ("not good"), sarcasm or context; the data is a random sample of the full dataset; reviews are English only.

## AI use

AI-generated code is marked with `# AI-generated` comments in the notebooks. The report notes which text was AI-drafted and which was only AI-reviewed.
