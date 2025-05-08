
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    confusion_matrix,
)


SEED      = 42
CSV_PATH  = "data_Enhanced_Imputed.csv"
SAVE_MDL  = "svm_progression_full.joblib"
SAVE_CM   = "svm_confusion_cv.png"

# ───────── 1. load ─────────
df = pd.read_csv(CSV_PATH)
if "Label_Min" not in df.columns:
    raise ValueError("cannot find 'Label_Min'。")

y = df["Label_Min"].astype(int)
drop_cols = ["SubjectID", "Label_Min", "Label_Max"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])



# ───────── 2. pipeline ─────────
pipe = Pipeline([
    ("imp",    SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
    ("svm",    SVC(kernel="rbf", class_weight="balanced"))
])

param_grid = {
    "svm__C":     [0.5, 1, 5, 10, 50],
    "svm__gamma": ["scale", 0.01, 0.005, 0.001]
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=2
)

# ───────── 3. eval ─────────
grid.fit(X, y)

print("\n=== result ===")
print("Best params      :", grid.best_params_)
print(f"Best CV balACC   : {grid.best_score_:.3f}")

y_pred_cv = cross_val_predict(
    grid.best_estimator_, X, y,
    cv=cv, n_jobs=-1
)

print("\n=== Cross-Val Classification Report ===")
print(classification_report(y, y_pred_cv, digits=3))
print("Cross-Val balACC :", balanced_accuracy_score(y, y_pred_cv))
acc  = accuracy_score(y, y_pred_cv)
mae  = mean_absolute_error(y, y_pred_cv)
f1_w = f1_score(y, y_pred_cv, average="weighted")   # 或 "macro"

print("\n=== Cross-Val Metrics ===")
print(f"Accuracy            : {acc:.3f}")
print(f"MAE (ordinal)       : {mae:.3f}")
print(f"F1-score (weighted) : {f1_w:.3f}")

cm = confusion_matrix(y, y_pred_cv, labels=[1, 2, 3, 4])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig(SAVE_CM, dpi=300)
plt.close()
print(f"Confusion matrix saved to '{SAVE_CM}'")


best_svm = grid.best_estimator_
best_svm.fit(X, y)
dump(best_svm, SAVE_MDL)
print(f"Final model saved to '{SAVE_MDL}'")


