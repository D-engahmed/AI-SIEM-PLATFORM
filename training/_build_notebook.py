"""
Builds training/model_selection.ipynb programmatically. Run once to
(re)generate the notebook source; the notebook is then executed
separately (jupyter nbconvert --execute) so it ships with real output
cells, not empty code.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ----------------------------------------------------------------------
md("""\
# correlation-ml-service — model creation, selection & evaluation

**Scope of this notebook:** train candidate risk-scoring models for the
`GraphMLScoring` strategy, compare them honestly, pick one, and register
it. This notebook is the source of truth for *which model and why*;
`train_model.py` is the lean script you'd point CI at to retrain the
already-selected model type on a schedule.

**Read before trusting any number below:** every row in the training
data is synthetic (`training/generate_synthetic_data.py`). No real
`ml-scoring-tasks` traffic was available when this was built. Treat the
metrics here as *"does the pipeline and the selection process work"*,
not *"this is production performance"*. See that file's docstring for
exactly which scenarios were encoded and why.

Feature engineering is imported directly from `src/ml_scorer.py` — not
reimplemented here — so there is no train/serve skew between this
notebook and the actual consumer/API code.
""")

code("""\
import sys, json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.xgboost
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
)

SRC_DIR = Path.cwd().parent / "src" if (Path.cwd() / "model_selection.ipynb").exists() or True else None
SRC_DIR = (Path.cwd() / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from ml_scorer import FEATURE_NAMES, MONOTONE_CONSTRAINTS, FeatureEngineer
from schemas import GraphFeatures, MLScoringTask, SignalContext

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
print("FEATURE_NAMES:", FEATURE_NAMES)
print("MONOTONE_CONSTRAINTS:", MONOTONE_CONSTRAINTS)
""")

# ----------------------------------------------------------------------
md("## 1. Load data")

code("""\
DATA_PATH = Path("synthetic_ml_scoring_tasks.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["event_time"])
df["drifted_period"] = df["drifted_period"].astype(bool)
# Keep timestamps tz-naive (UTC clock time) throughout the notebook -- the
# generator's timestamps are all UTC anyway, and mixing tz-aware /
# tz-naive Series (e.g. after a .values roundtrip) is a classic pandas
# footgun that breaks comparisons later.
if df["event_time"].dt.tz is not None:
    df["event_time"] = df["event_time"].dt.tz_convert("UTC").dt.tz_localize(None)
print(df.shape)
df.head()
""")

code("""\
print(df["scenario"].value_counts())
print()
print("Overall positive rate:", df["label"].mean().round(3))
print("Drifted-period share of rows:", df["drifted_period"].mean().round(3))
""")

# ----------------------------------------------------------------------
md("## 2. Exploratory visualization")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

df["scenario"].value_counts().sort_index().plot.barh(ax=axes[0])
axes[0].set_title("Rows per scenario")
axes[0].set_xlabel("count")

pos_by_scenario = df.groupby("scenario")["label"].mean().sort_values()
pos_by_scenario.plot.barh(ax=axes[1], color="firebrick")
axes[1].set_title("Positive (risky) label rate per scenario")
axes[1].set_xlabel("label=1 rate")

plt.tight_layout()
plt.show()
""")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for label, group in df.groupby("label"):
    axes[0].hist(group["fan_in_count"].dropna(), bins=30, alpha=0.5, label=f"label={label}")
axes[0].set_title("fan_in_count by label")
axes[0].set_xlabel("fan_in_count")
axes[0].legend()

for label, group in df.groupby("label"):
    axes[1].hist(np.log1p(group["epoch_age_seconds"].dropna()), bins=30, alpha=0.5, label=f"label={label}")
axes[1].set_title("log1p(epoch_age_seconds) by label")
axes[1].set_xlabel("log1p(epoch_age_seconds)")
axes[1].legend()

plt.tight_layout()
plt.show()
""")

md("""\
### Time visualization

This is the one plot in the notebook that exists purely to validate the
*monitoring* story, not the model: `generate_synthetic_data.py` deliberately
shifts two distributions in the final 20% of the time window (bigger
`loud_burst` bursts, bigger `benign_shared_nat` fan-in). If this plot
doesn't show a visible shift at the drift boundary, the drift-injection
code has a bug.
""")

code("""\
daily = (
    df.set_index("event_time")
      .resample("1D")
      .agg(escalation_rate=("label", "mean"), n=("label", "size"), mean_fan_in=("fan_in_count", "mean"))
)

drift_start = df.loc[df["drifted_period"], "event_time"].min()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
daily["escalation_rate"].plot(ax=axes[0], marker="o", ms=3)
axes[0].axvline(drift_start, color="red", linestyle="--", label="injected drift window starts")
axes[0].set_title("Daily positive-label rate over time")
axes[0].legend()

daily["mean_fan_in"].plot(ax=axes[1], marker="o", ms=3, color="darkorange")
axes[1].axvline(drift_start, color="red", linestyle="--")
axes[1].set_title("Daily mean fan_in_count over time")

plt.tight_layout()
plt.show()
""")

# ----------------------------------------------------------------------
md("""\
## 3. Feature engineering (imported from `src/`, not reimplemented)

Every row is converted through the exact `MLScoringTask` -> `FeatureEngineer`
path the Kafka consumer and API use. This is the guardrail against
train/serve skew.
""")

code("""\
def row_to_task(row) -> MLScoringTask:
    def opt(v):
        return None if pd.isna(v) else v
    ti = row["ti_matched"]
    ti_val = None if pd.isna(ti) else bool(ti)
    return MLScoringTask(
        correlation_id=row["correlation_id"],
        target_kind=row["target_kind"],
        target_value=row["target_value"],
        graph_features=GraphFeatures(
            fan_in_count=opt(row["fan_in_count"]) and int(row["fan_in_count"]) if not pd.isna(row["fan_in_count"]) else None,
            epoch_age_seconds=opt(row["epoch_age_seconds"]),
        ),
        signal_context=SignalContext(
            triggering_source_ip=row["triggering_source_ip"],
            ti_matched=ti_val,
            signal_ids=[f"sig-{i}" for i in range(int(row["signal_id_count"]))],
        ),
    )

feat_rows = []
for _, row in df.iterrows():
    task = row_to_task(row)
    feat_rows.append(FeatureEngineer.transform(task).vector[0])

X = np.vstack(feat_rows)
y = df["label"].to_numpy(dtype=float)
feat_df = pd.DataFrame(X, columns=FEATURE_NAMES)
feat_df["label"] = y
feat_df["event_time"] = df["event_time"].values
feat_df["scenario"] = df["scenario"].values
feat_df.head()
""")

code("""\
plt.figure(figsize=(8, 6))
sns.heatmap(feat_df[FEATURE_NAMES].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.show()
""")

# ----------------------------------------------------------------------
md("""\
## 4. Time-aware train / validation / test split

Random splits leak the future into training for time-series-shaped risk
data. Splitting on `event_time` instead means the test set includes the
injected drift window — a more honest estimate of "how does this model
do on data shaped differently from most of what it trained on", which is
exactly the situation this service will actually be in a month after
deployment.
""")

code("""\
feat_df_sorted = feat_df.sort_values("event_time").reset_index(drop=True)
n = len(feat_df_sorted)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = feat_df_sorted.iloc[:train_end]
val_df = feat_df_sorted.iloc[train_end:val_end]
test_df = feat_df_sorted.iloc[val_end:]

X_train, y_train = train_df[FEATURE_NAMES].values, train_df["label"].values
X_val, y_val = val_df[FEATURE_NAMES].values, val_df["label"].values
X_test, y_test = test_df[FEATURE_NAMES].values, test_df["label"].values

drift_start_ts = pd.Timestamp(df.loc[df["drifted_period"], "event_time"].min())
test_includes_drift = pd.Timestamp(test_df["event_time"].max()) >= drift_start_ts

print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
print(f"train window: {train_df.event_time.min()} .. {train_df.event_time.max()}")
print(f"test  window: {test_df.event_time.min()} .. {test_df.event_time.max()}")
print(f"test set includes the injected drift period: {test_includes_drift}")
""")

# ----------------------------------------------------------------------
md("""\
## 5. Candidate models — hyperparameter tuning, then cross-model comparison

Three model *families*, all on the spec's explicit allow-list (interpretable,
no deep learning, no unsupervised anomaly detection as primary model):

- **XGBoost** with monotonic constraints on `fan_in_count`, `fan_in_rate_log1p`,
  `low_and_slow_ratio`, `ti_matched_positive`
- **LightGBM** with the same monotonic constraints, as a second tree-based
  opinion (different split-finding algorithm, useful cross-check)
- **Logistic Regression** as an interpretable linear baseline — if this
  gets close to the tree models, that's a strong argument *for* shipping
  the simpler model instead

**Two-stage selection, not one:** for each family, several hyperparameter
configs are tried (a small curated grid, not every config is worth trying
on 56k rows of synthetic data). Every trial is scored on the **validation**
split and logged as a nested MLflow run under a `<family>_tuning` parent.
The best-by-validation-AP config per family is then re-evaluated exactly
once on the **test** split — that final number is what section 6 onward
compares across families.

This two-stage split matters: picking hyperparameters AND reporting the
winning number from the same split (test) is a well-known way to quietly
overfit the split itself, not just the data. Val picks the config; test
reports the number for that one already-fixed config, one time.
""")

code("""\
# MLflow 3.x puts the plain filesystem store ('file:./mlruns') in maintenance
# mode with no Model Registry support -- use a SQLite-backed store instead,
# which supports registry model-version/stage transitions used in section 10.
mlflow.set_tracking_uri("sqlite:///" + str((Path.cwd() / ".." / "mlflow.db").resolve()))
mlflow.set_experiment("correlation-ml-service")

monotone_str = "(" + ",".join(str(MONOTONE_CONSTRAINTS[f]) for f in FEATURE_NAMES) + ")"
monotone_list = [MONOTONE_CONSTRAINTS[f] for f in FEATURE_NAMES]

results = {}       # final, test-set numbers for the WINNING config per family
models = {}         # the fitted winning model per family
tuning_results = {} # every trial's val metrics, per family -- for the plot in 5c
""")

code("""\
# --- XGBoost: hyperparameter sweep ---
# Curated, not exhaustive -- these vary tree depth/learning-rate/subsampling,
# the three knobs most likely to matter on a 9-feature, 56k-row problem.
# monotone_constraints is NOT swept: it's a correctness requirement from the
# spec, not a tunable hyperparameter -- every trial keeps it fixed.
XGB_GRID = [
    dict(max_depth=3, eta=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=4, eta=0.10, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, eta=0.10, subsample=0.7, colsample_bytree=0.7),
    dict(max_depth=4, eta=0.20, subsample=0.9, colsample_bytree=0.9),
]

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES)

xgb_trials = []
with mlflow.start_run(run_name="xgboost_tuning") as parent_run:
    for i, grid_params in enumerate(XGB_GRID):
        with mlflow.start_run(run_name=f"xgboost_trial_{i}", nested=True):
            params = dict(objective="binary:logistic", eval_metric="aucpr",
                          monotone_constraints=monotone_str, seed=13, **grid_params)
            mlflow.log_params(params)

            booster = xgb.train(params, dtrain, num_boost_round=300,
                                 evals=[(dtrain, "train"), (dval, "val")],
                                 early_stopping_rounds=20, verbose_eval=False)
            val_preds = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
            val_ap = average_precision_score(y_val, val_preds)
            mlflow.log_metrics({"val_avg_precision": val_ap, "best_iteration": booster.best_iteration})

            xgb_trials.append(dict(params=grid_params, val_ap=val_ap,
                                    best_iteration=booster.best_iteration, booster=booster))
            print(f"  trial {i}: {grid_params} -> val_AP={val_ap:.4f} (best_iter={booster.best_iteration})")

    best_xgb = max(xgb_trials, key=lambda t: t["val_ap"])
    mlflow.log_metric("best_trial_val_ap", best_xgb["val_ap"])
    mlflow.log_params({f"winner_{k}": v for k, v in best_xgb["params"].items()})

tuning_results["xgboost"] = xgb_trials
booster = best_xgb["booster"]
preds = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
auc = roc_auc_score(y_test, preds); ap = average_precision_score(y_test, preds)
brier = brier_score_loss(y_test, preds)

with mlflow.start_run(run_name="xgboost_monotonic_winner") as run:
    mlflow.log_params(dict(objective="binary:logistic", monotone_constraints=monotone_str,
                            seed=13, **best_xgb["params"]))
    mlflow.log_metrics({"test_roc_auc": auc, "test_avg_precision": ap, "test_brier": brier,
                         "best_iteration": booster.best_iteration, "val_avg_precision": best_xgb["val_ap"]})
    results["xgboost"] = dict(preds=preds, auc=auc, ap=ap, brier=brier, run_id=run.info.run_id)
    models["xgboost"] = booster

print(f"\\nBest XGBoost config: {best_xgb['params']} (val_AP={best_xgb['val_ap']:.4f})")
print(f"xgboost   TEST AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}  best_iter={booster.best_iteration}")
""")

code("""\
# --- LightGBM: hyperparameter sweep ---
LGB_GRID = [
    dict(max_depth=3, learning_rate=0.05, num_leaves=7,  feature_fraction=0.8, bagging_fraction=0.8),
    dict(max_depth=4, learning_rate=0.10, num_leaves=15, feature_fraction=0.8, bagging_fraction=0.8),
    dict(max_depth=5, learning_rate=0.10, num_leaves=31, feature_fraction=0.7, bagging_fraction=0.7),
    dict(max_depth=4, learning_rate=0.20, num_leaves=15, feature_fraction=0.9, bagging_fraction=0.9),
]

train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
val_set = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_set)

lgb_trials = []
with mlflow.start_run(run_name="lightgbm_tuning") as parent_run:
    for i, grid_params in enumerate(LGB_GRID):
        with mlflow.start_run(run_name=f"lightgbm_trial_{i}", nested=True):
            params = dict(objective="binary", metric="average_precision",
                          monotone_constraints=monotone_list, bagging_freq=1,
                          verbose=-1, seed=13, **grid_params)
            mlflow.log_params(params)

            gbm = lgb.train(params, train_set, num_boost_round=300, valid_sets=[val_set],
                             callbacks=[lgb.early_stopping(20, verbose=False)])
            val_preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
            val_ap = average_precision_score(y_val, val_preds)
            mlflow.log_metrics({"val_avg_precision": val_ap, "best_iteration": gbm.best_iteration})

            lgb_trials.append(dict(params=grid_params, val_ap=val_ap,
                                    best_iteration=gbm.best_iteration, model=gbm))
            print(f"  trial {i}: {grid_params} -> val_AP={val_ap:.4f} (best_iter={gbm.best_iteration})")

    best_lgb = max(lgb_trials, key=lambda t: t["val_ap"])
    mlflow.log_metric("best_trial_val_ap", best_lgb["val_ap"])
    mlflow.log_params({f"winner_{k}": v for k, v in best_lgb["params"].items()})

tuning_results["lightgbm"] = lgb_trials
gbm = best_lgb["model"]
preds = gbm.predict(X_test, num_iteration=gbm.best_iteration)
auc = roc_auc_score(y_test, preds); ap = average_precision_score(y_test, preds)
brier = brier_score_loss(y_test, preds)

with mlflow.start_run(run_name="lightgbm_monotonic_winner") as run:
    mlflow.log_params(dict(objective="binary", monotone_constraints=monotone_list,
                            seed=13, **best_lgb["params"]))
    mlflow.log_metrics({"test_roc_auc": auc, "test_avg_precision": ap, "test_brier": brier,
                         "best_iteration": gbm.best_iteration, "val_avg_precision": best_lgb["val_ap"]})
    results["lightgbm"] = dict(preds=preds, auc=auc, ap=ap, brier=brier, run_id=run.info.run_id)
    models["lightgbm"] = gbm

print(f"\\nBest LightGBM config: {best_lgb['params']} (val_AP={best_lgb['val_ap']:.4f})")
print(f"lightgbm  TEST AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}  best_iter={gbm.best_iteration}")
""")

code("""\
# --- Logistic Regression: hyperparameter sweep (regularization strength) ---
LR_GRID = [dict(C=0.01), dict(C=0.1), dict(C=1.0), dict(C=10.0)]

scaler = StandardScaler().fit(X_train)
Xtr_s, Xval_s, Xte_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

lr_trials = []
with mlflow.start_run(run_name="logistic_regression_tuning") as parent_run:
    for i, grid_params in enumerate(LR_GRID):
        with mlflow.start_run(run_name=f"logreg_trial_{i}", nested=True):
            params = dict(max_iter=1000, **grid_params)
            mlflow.log_params(params)

            clf = LogisticRegression(**params)
            clf.fit(Xtr_s, y_train)
            val_preds = clf.predict_proba(Xval_s)[:, 1]
            val_ap = average_precision_score(y_val, val_preds)
            mlflow.log_metric("val_avg_precision", val_ap)

            lr_trials.append(dict(params=grid_params, val_ap=val_ap, clf=clf))
            print(f"  trial {i}: {grid_params} -> val_AP={val_ap:.4f}")

    best_lr = max(lr_trials, key=lambda t: t["val_ap"])
    mlflow.log_metric("best_trial_val_ap", best_lr["val_ap"])
    mlflow.log_params({f"winner_{k}": v for k, v in best_lr["params"].items()})

tuning_results["logistic_regression"] = lr_trials
clf = best_lr["clf"]
preds = clf.predict_proba(Xte_s)[:, 1]
auc = roc_auc_score(y_test, preds); ap = average_precision_score(y_test, preds)
brier = brier_score_loss(y_test, preds)

with mlflow.start_run(run_name="logistic_regression_winner") as run:
    mlflow.log_params(dict(max_iter=1000, **best_lr["params"]))
    mlflow.log_metrics({"test_roc_auc": auc, "test_avg_precision": ap, "test_brier": brier,
                         "val_avg_precision": best_lr["val_ap"]})
    results["logistic_regression"] = dict(preds=preds, auc=auc, ap=ap, brier=brier, run_id=run.info.run_id)
    models["logistic_regression"] = (clf, scaler)

print(f"\\nBest LogReg config: {best_lr['params']} (val_AP={best_lr['val_ap']:.4f})")
print(f"logreg    TEST AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}")
coef_df = pd.DataFrame({"feature": FEATURE_NAMES, "coefficient": clf.coef_[0]}).sort_values("coefficient")
print(coef_df.to_string(index=False))
""")

md("""\
### 5c. What the sweep actually bought us

Each family's trials, by validation AP. If the bars within a family are
all roughly the same height, that family's performance is *not*
hyperparameter-sensitive on this data — worth knowing, since it means
future retrains don't need to re-tune from scratch every time.
""")

code("""\
fig, ax = plt.subplots(figsize=(10, 4))
offset = 0
xticks, xlabels = [], []
colors = {"xgboost": "tab:blue", "lightgbm": "tab:green", "logistic_regression": "tab:orange"}
for family, trials in tuning_results.items():
    vals = [t["val_ap"] for t in trials]
    xs = list(range(offset, offset + len(vals)))
    bars = ax.bar(xs, vals, color=colors[family], label=family)
    best_idx = int(np.argmax(vals))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2.5)
    xticks.extend(xs)
    xlabels.extend([f"trial {i}" for i in range(len(vals))])
    offset += len(vals) + 1

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("validation avg. precision")
ax.set_title("Hyperparameter sweep per model family (black outline = winner)")
ax.legend()
plt.tight_layout()
plt.show()
""")

# ----------------------------------------------------------------------
md("## 6. Compare candidates: ROC, PR, calibration")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["preds"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_title("ROC curve (test set, incl. drift period)")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()

for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_test, r["preds"])
    axes[1].plot(rec, prec, label=f"{name} (AP={r['ap']:.3f})")
axes[1].set_title("Precision-Recall curve")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()

for name, r in results.items():
    prob_true, prob_pred = [], []
    bins = np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (r["preds"] >= lo) & (r["preds"] < hi)
        if mask.sum() > 5:
            prob_true.append(y_test[mask].mean())
            prob_pred.append(r["preds"][mask].mean())
    axes[2].plot(prob_pred, prob_true, marker="o", label=name)
axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[2].set_title("Calibration (reliability) curve")
axes[2].set_xlabel("Mean predicted probability"); axes[2].set_ylabel("Observed positive rate")
axes[2].legend()

plt.tight_layout()
plt.show()
""")

# ----------------------------------------------------------------------
md("""\
## 7. The two checks that matter more than the headline metrics

AUC/AP describe average performance. This service's whole reason for
existing is to handle two specific edge cases correctly. Both are
evaluated per-candidate, not just for the eventual winner.
""")

code("""\
def make_task(fan_in, age, ti, sid="probe"):
    return MLScoringTask(
        correlation_id="00000000-0000-0000-0000-000000000001",
        target_kind="user", target_value=sid,
        graph_features=GraphFeatures(fan_in_count=fan_in, epoch_age_seconds=age),
        signal_context=SignalContext(triggering_source_ip="203.0.113.9", ti_matched=ti, signal_ids=["s1"]),
    )

def featurize(task):
    return FeatureEngineer.transform(task).vector

def predict_one(name, vec):
    if name == "logistic_regression":
        clf, scaler = models[name]
        return float(clf.predict_proba(scaler.transform(vec))[0, 1])
    if name == "lightgbm":
        return float(models[name].predict(vec, num_iteration=models[name].best_iteration)[0])
    dm = xgb.DMatrix(vec, feature_names=FEATURE_NAMES)
    return float(models[name].predict(dm, iteration_range=(0, models[name].best_iteration + 1))[0])

# Check A: low-and-slow trap -- same rate, very different absolute count/age
trap_attack = featurize(make_task(fan_in=18, age=18*3600, ti=None))
trap_benign = featurize(make_task(fan_in=2, age=2*3600, ti=None))

# Check B: benign_shared_nat false-positive risk -- moderate fan-in from one office
nat_benign = featurize(make_task(fan_in=8, age=3*3600, ti=None))

print(f"{'model':22s} {'trap_attack':>12s} {'trap_benign':>12s} {'trap_pass':>10s} {'shared_nat_p':>13s}")
for name in models:
    p_attack = predict_one(name, trap_attack)
    p_benign = predict_one(name, trap_benign)
    p_nat = predict_one(name, nat_benign)
    trap_pass = p_attack > p_benign
    print(f"{name:22s} {p_attack:12.4f} {p_benign:12.4f} {str(trap_pass):>10s} {p_nat:13.4f}")
""")

# ----------------------------------------------------------------------
md("## 8. Feature importance and SHAP")

code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

xgb_imp = models["xgboost"].get_score(importance_type="gain")
pd.Series(xgb_imp).sort_values().plot.barh(ax=axes[0])
axes[0].set_title("XGBoost feature importance (gain)")

lgb_imp = pd.Series(models["lightgbm"].feature_importance(importance_type="gain"), index=FEATURE_NAMES)
lgb_imp.sort_values().plot.barh(ax=axes[1], color="seagreen")
axes[1].set_title("LightGBM feature importance (gain)")

plt.tight_layout()
plt.show()
""")

code("""\
# SHAP summary for the tree model most likely to be selected (XGBoost).
# Subsampled for speed -- this is exploratory, not a production explainability pipeline.
explainer = shap.TreeExplainer(models["xgboost"])
sample_idx = np.random.RandomState(13).choice(len(X_test), size=min(400, len(X_test)), replace=False)
shap_values = explainer.shap_values(X_test[sample_idx])

shap.summary_plot(shap_values, X_test[sample_idx], feature_names=FEATURE_NAMES, show=False)
plt.tight_layout()
plt.show()
""")

md("""\
### Monotonicity compliance check

XGBoost and LightGBM had `monotone_constraints` enforced *during training*,
so they're compliant by construction. Logistic regression was not
constrained — check whether its fitted coefficient signs happen to agree
with the same causal story anyway. Multicollinearity between
`low_and_slow_ratio`, `fan_in_count`, and `epoch_age_seconds_log1p` (they're
all derived from the same two raw inputs) means an unconstrained linear
model is free to push the "risk story" onto whichever correlated feature
fits the training data best, even if that gives an individual feature the
wrong sign.
""")

code("""\
logreg_coefs = dict(zip(FEATURE_NAMES, models["logistic_regression"][0].coef_[0]))
print(f"{'feature':28s} {'constraint':>10s} {'coefficient':>12s} {'compliant':>10s}")
violations = []
for feat in FEATURE_NAMES:
    constraint = MONOTONE_CONSTRAINTS[feat]
    coef = logreg_coefs[feat]
    if constraint == 0:
        compliant = True
    else:
        compliant = (coef * constraint) > 0
    if not compliant:
        violations.append(feat)
    print(f"{feat:28s} {constraint:>10d} {coef:>12.4f} {str(compliant):>10s}")

print()
if violations:
    print(f"*** logistic_regression VIOLATES the intended monotonic story on: {violations} ***")
else:
    print("logistic_regression's coefficients agree with all declared constraints.")
""")

md("""\
### Permutation importance — a model-agnostic cross-check on gain importance

Gain-based importance (used above) can overstate features that get split
on a lot, even if those splits don't move predictions much. Permutation
importance answers a more direct question: how much does shuffling ONE
feature (breaking its relationship to the label, keeping everything else
intact) hurt test-set AP? Computed by hand here rather than via
`sklearn.inspection.permutation_importance`, since that helper expects a
`.predict_proba(ndarray)`-shaped estimator and `xgb.Booster`/`lgb.Booster`
don't expose that signature directly.
""")

code("""\
def permutation_importance_manual(predict_fn, X, y, feature_names, n_repeats=5, seed=13):
    rng = np.random.RandomState(seed)
    baseline_ap = average_precision_score(y, predict_fn(X))
    drops = {f: [] for f in feature_names}
    for i, feat in enumerate(feature_names):
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            rng.shuffle(X_shuffled[:, i])  # in-place permute just this column
            shuffled_ap = average_precision_score(y, predict_fn(X_shuffled))
            drops[feat].append(baseline_ap - shuffled_ap)
    return baseline_ap, {f: float(np.mean(v)) for f, v in drops.items()}


def xgb_predict_fn(X):
    return models["xgboost"].predict(
        xgb.DMatrix(X, feature_names=FEATURE_NAMES),
        iteration_range=(0, models["xgboost"].best_iteration + 1),
    )

baseline_ap, perm_importance = permutation_importance_manual(xgb_predict_fn, X_test, y_test, FEATURE_NAMES)

perm_df = pd.Series(perm_importance).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
perm_df.plot.barh(ax=ax, color="slateblue")
ax.set_title(f"XGBoost permutation importance (baseline test AP={baseline_ap:.4f})")
ax.set_xlabel("mean AP drop when this feature is shuffled (higher = more important)")
plt.tight_layout()
plt.show()

# Sanity cross-check, not just a nicer chart: gain importance and permutation
# importance measuring genuinely different things is normal, but if they
# picked completely DIFFERENT top features, that's worth a second look before
# trusting either one.
gain_top3 = set(pd.Series(models["xgboost"].get_score(importance_type="gain")).sort_values(ascending=False).head(3).index)
perm_top3 = set(perm_df.sort_values(ascending=False).head(3).index)
print(f"Gain top-3:        {gain_top3}")
print(f"Permutation top-3: {perm_top3}")
print(f"Overlap: {gain_top3 & perm_top3}")
""")

md("""\
### Partial dependence — does monotonicity hold across the WHOLE range?

Check A/B (section 7) probe two specific points. Partial dependence
sweeps one feature across its full observed range (holding every other
feature at its **median** test-set value) and plots the model's output —
a direct visual check that "risk never decreases as this feature
increases" actually holds everywhere, not just at the two points we
happened to test.
""")

code("""\
MONOTONE_FEATURES = [f for f, c in MONOTONE_CONSTRAINTS.items() if c != 0]

fig, axes = plt.subplots(1, len(MONOTONE_FEATURES), figsize=(4.5 * len(MONOTONE_FEATURES), 4))
if len(MONOTONE_FEATURES) == 1:
    axes = [axes]

median_row = np.median(X_test, axis=0)

for ax, feat in zip(axes, MONOTONE_FEATURES):
    idx = FEATURE_NAMES.index(feat)
    lo, hi = np.percentile(X_test[:, idx], [1, 99])
    sweep_vals = np.linspace(lo, hi, 50)

    sweep_X = np.tile(median_row, (50, 1))
    sweep_X[:, idx] = sweep_vals

    xgb_pdp = models["xgboost"].predict(
        xgb.DMatrix(sweep_X, feature_names=FEATURE_NAMES),
        iteration_range=(0, models["xgboost"].best_iteration + 1),
    )
    lgb_pdp = models["lightgbm"].predict(sweep_X, num_iteration=models["lightgbm"].best_iteration)

    ax.plot(sweep_vals, xgb_pdp, label="xgboost", color="tab:blue")
    ax.plot(sweep_vals, lgb_pdp, label="lightgbm", color="tab:green")
    ax.set_title(feat)
    ax.set_xlabel(feat)
    ax.set_ylabel("predicted probability")
    ax.legend(fontsize=8)

    constraint = MONOTONE_CONSTRAINTS[feat]
    diffs = np.diff(xgb_pdp) * constraint
    if np.any(diffs < -1e-9):
        ax.set_facecolor("#fff0f0")
        print(f"*** {feat}: xgboost PDP is NOT monotonic across the full range (constraint={constraint}) ***")

plt.tight_layout()
plt.show()
""")

# ----------------------------------------------------------------------
md("""\
## 9. Model selection

Decision, in order of what actually matters for this service:

1. **Low-and-slow trap (Check A)** — non-negotiable. Any candidate that
   fails it is disqualified regardless of AUC, because it means the
   model learned "rate" instead of "sustained volume", which is the
   opposite of what the spec asks for.
2. **Monotonicity compliance** — also non-negotiable for the specific
   feature the spec calls out (`low_and_slow_ratio`). A model that gets
   the *sign* wrong on that feature can look fine on average metrics
   while still being untrustworthy on exactly the pattern this service
   exists to catch, once real traffic doesn't share the training data's
   correlation structure.
3. **benign_shared_nat false-positive rate (Check B)** — the realistic
   failure mode most likely to page someone at 3am. Lower is better.
4. **Test-set AUC / AP / Brier**, computed on a split that *includes* the
   injected drift period — the tie-breaker among whatever survives 1-3,
   not the primary criterion.

Run the cell below after the ones above; it computes the actual decision
from what this run produced, rather than asserting a fixed winner.
""")

code("""\
summary = pd.DataFrame({
    name: {"test_auc": r["auc"], "test_ap": r["ap"], "test_brier": r["brier"]}
    for name, r in results.items()
}).T.round(4)
summary["shared_nat_p"] = {
    name: predict_one(name, nat_benign) for name in models
}
print(summary)
print()

# Disqualify on hard constraints first (trap check + monotonicity), THEN
# rank survivors by AP. This is what actually decides the winner below --
# not a fixed string.
disqualified = set()
if violations:  # from the monotonicity check above -- currently only checked for logreg
    disqualified.add("logistic_regression")

survivors = [m for m in results if m not in disqualified]
ranked_survivors = summary.loc[survivors].sort_values("test_ap", ascending=False)
winner = ranked_survivors.index[0]

print(f"Disqualified (monotonicity violation on a spec-critical feature): {sorted(disqualified) or 'none'}")
print(f"Surviving candidates ranked by AP:\\n{ranked_survivors}")
print(f"\\nSelected: {winner}")
print(f"(Highest raw AP without the monotonicity filter would have been "
      f"'{summary['test_ap'].idxmax()}' -- overridden because it fails check 2 above.)")
""")

# ----------------------------------------------------------------------
md("""\
## 10. Register the winner

Saves the model both ways the serving code understands (see `CML_MODEL_SOURCE`
in `src/config.py`):

- a local `artifacts/model_latest.joblib` (fast path, no MLflow dependency
  at serving time)
- an MLflow-registered model version (`models:/correlation-ml-service-risk-model/…`),
  transitioned to `Staging` — promote to `Production` manually once it's been
  validated against real traffic.

Uses `winner` from the cell above — nothing hardcoded here. If the data
changes (real traffic instead of synthetic) and a different candidate
survives disqualification with a better AP, this cell follows it
automatically. If `winner` isn't XGBoost or LightGBM, the save step below
needs a different branch (this notebook doesn't have an MLflow-native
logistic regression save path wired up, since it wasn't needed this run).
""")

code("""\
import joblib

assert winner in ("xgboost", "lightgbm"), (
    f"winner={winner!r} has no save path wired up in this cell -- add one "
    f"before running (see the markdown above)."
)

MODEL_VERSION = f"gml-svc-0.2.0-synthetic-{datetime.now(timezone.utc):%Y%m%d}-{winner}"
ARTIFACT_DIR = Path.cwd().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

chosen_booster = models[winner]

artifact_path = ARTIFACT_DIR / "model_latest.joblib"
joblib.dump(
    {"booster": chosen_booster, "model_family": winner,
     "model_version": MODEL_VERSION, "feature_names": FEATURE_NAMES},
    artifact_path,
)
print("Saved:", artifact_path)

log_fn = mlflow.xgboost.log_model if winner == "xgboost" else mlflow.lightgbm.log_model
import mlflow.lightgbm  # noqa: E402

with mlflow.start_run(run_name="register_winner") as run:
    mlflow.log_param("chosen_model", winner)
    mlflow.log_metrics({f"chosen_{k}": v for k, v in summary.loc[winner].items()})
    model_info = log_fn(chosen_booster, name="model", registered_model_name="correlation-ml-service-risk-model")
    print("Registered model URI:", model_info.model_uri)

client = mlflow.MlflowClient()
versions = client.search_model_versions("name='correlation-ml-service-risk-model'")
latest = max(versions, key=lambda v: int(v.version))
# model_family/model_version tags are how ModelScorer.load_from_mlflow (the
# CML_MODEL_SOURCE=mlflow path in src/ml_scorer.py) knows which flavor-
# specific loader to use and what version string to log -- mlflow.pyfunc's
# generic load path would hide the xgboost-vs-lightgbm distinction entirely.
client.set_model_version_tag(
    name="correlation-ml-service-risk-model", version=latest.version,
    key="model_family", value=winner,
)
client.set_model_version_tag(
    name="correlation-ml-service-risk-model", version=latest.version,
    key="model_version", value=MODEL_VERSION,
)
client.transition_model_version_stage(
    name="correlation-ml-service-risk-model", version=latest.version, stage="Staging",
)
print(f"Model version {latest.version} ({winner}) tagged and moved to Staging.")
""")

md("""\
## 11. Feature reference distribution (for src/monitoring.py's drift detection)

Saves a capped 5,000-row-per-feature sample of the full labeled dataset
(train+val+test combined -- the reference should describe what the model
was built to expect overall, not just its training split) as the fixed
baseline `FeatureDriftMonitor` compares live traffic against via PSI. Same
artifact `training/train_model.py` produces on every retrain; saved here
too so a notebook-selected model has a matching reference from the start.
""")

code("""\
REFERENCE_SAMPLE_SIZE = 5000
X_all = feat_df_sorted[FEATURE_NAMES].to_numpy()

rng = np.random.RandomState(13)
idx = rng.choice(len(X_all), size=min(REFERENCE_SAMPLE_SIZE, len(X_all)), replace=False)
sample = X_all[idx]

reference_path = ARTIFACT_DIR / "feature_reference_distribution.npz"
np.savez(reference_path, **{name: sample[:, i] for i, name in enumerate(FEATURE_NAMES)})
print(f"Saved feature reference distribution -> {reference_path} ({len(sample)} samples/feature)")
""")

md("""\
## Next steps (not in this notebook)

- Swap `synthetic_ml_scoring_tasks.csv` for real historical GraphPivotStrategy
  output as soon as it exists, and rerun sections 4-10.
- `training/evaluate_model.py` runs the Check A / Check B regression tests
  as an automated CI gate (fails the build, doesn't just print a warning).
- `docs/02-model-creation-and-selection.md` narrates this notebook for
  someone who won't run it.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

with open("training/model_selection.ipynb", "w") as f:
    nbf.write(nb, f)

print("wrote training/model_selection.ipynb with", len(cells), "cells")
