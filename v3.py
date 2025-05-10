import os, re, json, time, warnings, joblib, optuna
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

CSV_PATH, MODEL_DIR = "survey.csv", "models"
SEED                = 42
N_TRIALS            = 128
CV_FOLDS            = 5
N_JOBS              = 8
TOP_K               = 2
HOLDOUT_FRAC        = 0.20
SQLITE_DB           = "osmi_optuna.db"
LOCK_PREFIX         = "/tmp/osmi_gpu-"
USE_XGB             = True
USE_RF              = True
DROP_LOCATION       = True

XGB_PARAMS = dict(
    learning_rate      = 0.05,
    n_estimators       = 400,
    max_depth          = 4,
    min_child_weight   = 6,
    subsample          = 0.70,
    colsample_bytree   = 0.70,
    reg_lambda         = 10.0,
    reg_alpha          = 4.0,
    objective          = "multi:softprob",
    tree_method        = "gpu_hist",
    predictor          = "gpu_predictor",
    enable_categorical = True,
    num_class          = 3,
    random_state       = SEED,
    gpu_id             = 0,
)

RF_PARAMS = dict(
    n_estimators      = 357,
    max_depth         = 7,
    min_samples_split = 6,
    min_samples_leaf  = 6,
    max_features      = "sqrt",
    bootstrap         = True,
    criterion         = "entropy",
    n_jobs            = -1,
    random_state      = SEED,
)

_GPU_IDS = (os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if os.environ.get("CUDA_VISIBLE_DEVICES")
            else [str(i) for i in range(8)])
_NUM_GPUS = len(_GPU_IDS)

class GPULock:
    def __init__(self, gpu_id, retry=2.0):
        self.path = f"{LOCK_PREFIX}{gpu_id}.lock"; self.retry = retry; self.fp = None
    def __enter__(self):
        while True:
            try:
                self.fp = os.open(self.path, os.O_CREAT|os.O_EXCL|os.O_RDWR)
                os.write(self.fp, str(os.getpid()).encode()); return
            except FileExistsError: time.sleep(self.retry)
    def __exit__(self, *a):
        if self.fp: os.close(self.fp); os.unlink(self.path)


pd.options.mode.copy_on_write = True
warnings.filterwarnings("ignore", category=FutureWarning)
_COMMENT_RE = re.compile(r"comment|^q[0-9]+_other$", re.I)

def load_raw(p): return pd.read_csv(p)

def basic_clean(df):
    df = df[df.Age.between(18, 100)].copy()
    df["Age"] = df["Age"].astype(int)

    g = df.Gender.astype(str).str.lower().str.strip()
    df.loc[g.str.match(r"^(m|male|cis[ -]?male)"), "Gender"] = "Male"
    df.loc[g.str.match(r"^(f|female|cis[ -]?female)"), "Gender"] = "Female"
    df.loc[~g.str.match(r"^(m|male|cis[ -]?male|f|female|cis[ -]?female)"),
           "Gender"] = "Other"

    if "Timestamp" in df.columns:
        ts = pd.to_datetime(df.Timestamp, errors="coerce")
        df["survey_year"], df["survey_month"] = ts.dt.year, ts.dt.month
        df.drop(columns="Timestamp", inplace=True)

    y = (df.work_interfere.astype(str).str.strip().str.title()
         .replace({"Nan": "Not Applicable", "Na": "Not Applicable"})
         .fillna("Not Applicable")
         .map({"Not Applicable": 0, "Never": 1,
               "Rarely": 1, "Sometimes": 2, "Often": 2})
         .fillna(0).astype(int).values)
    df.drop(columns="work_interfere", inplace=True)
    return df, y

def prune_features(df: pd.DataFrame) -> pd.DataFrame:
    """comment-cols, >50 % NA, constants, high-card cat, |ρ|>0.97 numerics"""
    df = df.drop(columns=[c for c in df.columns if _COMMENT_RE.search(c)])
    df = df.drop(columns=[c for c in df.columns if df[c].isna().mean() > 0.50])
    df = df.drop(columns=[c for c in df.columns if df[c].nunique(dropna=False) <= 1])
    df = df.drop(columns=[c for c in df.select_dtypes("object")
                          if df[c].nunique() > 200])
    num = df.select_dtypes(exclude="object")
    if not num.empty:
        corr = num.corr(numeric_only=True).abs()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        upper = corr.where(mask)
        to_drop = [c for c in upper.columns if (upper[c] > 0.97).any()]
        df = df.drop(columns=to_drop)
    return df

def sanitise(df):
    obj = df.select_dtypes("object").columns
    df[obj] = (df[obj].fillna("Unknown")
                       .astype("string[pyarrow]")
                       .astype("category"))
    num = df.columns.difference(obj)
    df[num] = df[num].fillna(-1)
    cat_idx = [df.columns.get_loc(c) for c in obj]
    return df, cat_idx

def encode_for_rf(df):
    df_rf = df.copy()
    for c in df_rf.select_dtypes("category"):
        df_rf[c] = df_rf[c].cat.codes.astype("int32")
    if DROP_LOCATION:
        df_rf = df_rf.drop(columns=["Country", "state"], errors="ignore")
    return df_rf


_CB_BASE = dict(
    loss_function="MultiClass", eval_metric="TotalF1",
    auto_class_weights="Balanced", bootstrap_type="Bayesian",
    iterations=3000, early_stopping_rounds=150,
    random_seed=SEED, task_type="GPU", gpu_ram_part=0.95,
    grow_policy="Lossguide", sampling_frequency="PerTree", verbose=False,
)
_SEARCH = dict(
    depth=(5, 10), learning_rate=(0.01, 0.25), l2_leaf_reg=(1, 10),
    bagging_temperature=(0.0, 1.0), border_count=[32, 64, 128, 254],
    random_strength=(0.1, 3.0),
)


def objective(trial):
    gpu_hint = _GPU_IDS[trial.number % _NUM_GPUS]
    for off in range(_NUM_GPUS):
        gpu = _GPU_IDS[(int(gpu_hint) + off) % _NUM_GPUS]
        with GPULock(gpu):
            params = dict(
                devices=gpu,
                depth              = trial.suggest_int   ("depth", *_SEARCH["depth"]),
                learning_rate      = trial.suggest_float ("learning_rate",
                                                           *_SEARCH["learning_rate"],
                                                           log=True),
                l2_leaf_reg        = trial.suggest_float ("l2_leaf_reg",
                                                           *_SEARCH["l2_leaf_reg"],
                                                           log=True),
                bagging_temperature= trial.suggest_float ("bagging_temperature",
                                                           *_SEARCH["bagging_temperature"]),
                border_count       = trial.suggest_categorical("border_count",
                                                               _SEARCH["border_count"]),
                random_strength    = trial.suggest_float ("random_strength",
                                                           *_SEARCH["random_strength"],
                                                           log=True),
            )
            skf = StratifiedKFold(n_splits=CV_FOLDS,
                                  shuffle=True, random_state=SEED)
            scores = []
            for tr, va in skf.split(_X, _y):
                cb = CatBoostClassifier(**(_CB_BASE | params))
                cb.fit(Pool(_X.iloc[tr], _y[tr], cat_features=_cat_idx))
                pred = cb.predict(_X.iloc[va],
                                  prediction_type="Class").astype(int).ravel()
                scores.append(f1_score(_y[va], pred, average="macro"))
            return float(np.mean(scores))

def build_ensemble(best_trials, X_tr, y_tr, cat_idx, X_rf_tr):
    models = []

    # CatBoosts
    for i, t in enumerate(best_trials, 1):
        gpu = _GPU_IDS[(i - 1) % _NUM_GPUS]
        with GPULock(gpu):
            cb = CatBoostClassifier(**(_CB_BASE | {"devices": gpu} | t.params))
            cb.fit(Pool(X_tr, y_tr, cat_features=cat_idx))
        models.append((f"cb{i}", cb, t.value))

    # XGBoost
    if USE_XGB:
        xgb = XGBClassifier(**XGB_PARAMS)
        xgb.fit(X_tr, y_tr)
        models.append(("xgb", xgb, float(np.mean([t.value for t in best_trials]))))

    # Random-Forest
    if USE_RF:
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_rf_tr, y_tr)
        models.append(("rf", rf, float(np.mean([t.value for t in best_trials]))))

    return models

def soft_vote(models, X_cb, X_rf):
    """Weighted soft vote; RF uses X_rf, others use X_cb."""
    weights = np.array([w for _, _, w in models])
    proba = np.zeros((len(X_cb), 3), dtype=np.float32)

    for name, mdl, w in models:
        if name.startswith("cb") or name == "xgb":
            proba += mdl.predict_proba(X_cb) * w
        elif name == "rf":
            proba += mdl.predict_proba(X_rf) * w

    proba /= weights.sum()
    return proba.argmax(axis=1)

def save_models(models):
    Path(MODEL_DIR).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {}
    for name, mdl, wt in models:
        fn = f"{name}_{ts}"
        if isinstance(mdl, CatBoostClassifier):
            p = Path(MODEL_DIR) / f"{fn}.cbm"
            mdl.save_model(p)
            meta[p.name] = {"type": "catboost", "weight": wt}
        elif isinstance(mdl, XGBClassifier):
            p = Path(MODEL_DIR) / f"{fn}.joblib"
            joblib.dump(mdl, p)
            meta[p.name] = {"type": "xgboost", "weight": wt}
        else:
            p = Path(MODEL_DIR) / f"{fn}.joblib"
            joblib.dump(mdl, p)
            meta[p.name] = {"type": "random_forest", "weight": wt}
    json.dump(meta, open(Path(MODEL_DIR) / f"ensemble_{ts}.json", "w"), indent=2)
    print("Artefacts saved →", MODEL_DIR)

def four_scores(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score   (y_true, y_pred, average="macro", zero_division=0),
            f1_score       (y_true, y_pred, average="macro"))

def main():
    global _X, _y, _cat_idx

    # ingest / clean
    df, _y = basic_clean(load_raw(CSV_PATH))
    df = prune_features(df)
    df, _cat_idx = sanitise(df)
    df.reset_index(drop=True, inplace=True)
    _X = df
    _X_rf = encode_for_rf(df)

    # Optuna search
    study = optuna.create_study(direction="maximize",
                                storage=f"sqlite:///{SQLITE_DB}",
                                study_name="osmi_gpu",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS,
                   n_jobs=N_JOBS, show_progress_bar=True)

    best = sorted(study.best_trials, key=lambda t: -t.value)[:TOP_K]
    for i, t in enumerate(best, 1):
        print(f"#{i}: F1={t.value:.3f} depth={t.params['depth']} "
              f"lr={t.params['learning_rate']:.3g}")

    tr, ho = train_test_split(np.arange(len(_y)),
                              stratify=_y,
                              test_size=HOLDOUT_FRAC,
                              random_state=SEED)

    models = build_ensemble(best,
                            _X.iloc[tr], _y[tr], _cat_idx,
                            _X_rf.iloc[tr])

    # Logistic Regression Baseline
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X_ohe = ohe.fit_transform(df)
    logreg = LogisticRegression(max_iter=2000,
                                solver="lbfgs",
                                multi_class="multinomial")
    logreg.fit(X_ohe[tr], _y[tr])
    models.append(("logreg", logreg, 0.0))

    headers = ["model", "split", "acc", "prec", "rec", "f1"]
    rows = []

    for name, mdl, _ in models:
        if name == "logreg":
            ytr_pred = mdl.predict(X_ohe[tr])
            yho_pred = mdl.predict(X_ohe[ho])
        elif name == "rf":
            ytr_pred = mdl.predict(_X_rf.iloc[tr])
            yho_pred = mdl.predict(_X_rf.iloc[ho])
        else:
            ytr_pred = mdl.predict(_X.iloc[tr])
            yho_pred = mdl.predict(_X.iloc[ho])
        rows.append([name, "train", *four_scores(_y[tr], ytr_pred)])
        rows.append([name, "test",  *four_scores(_y[ho], yho_pred)])

    # ensemble (CB + XGB + RF)
    ens_tr = soft_vote([m for m in models if m[0] not in ("logreg",)],
                       _X.iloc[tr], _X_rf.iloc[tr])
    ens_ho = soft_vote([m for m in models if m[0] not in ("logreg",)],
                       _X.iloc[ho], _X_rf.iloc[ho])
    rows.append(["ensemble", "train", *four_scores(_y[tr], ens_tr)])
    rows.append(["ensemble", "test",  *four_scores(_y[ho], ens_ho)])

    print("\n─ Performance Summary ─")
    print(f"{headers[0]:10} {headers[1]:5} {headers[2]:6} "
          f"{headers[3]:6} {headers[4]:6} {headers[5]:6}")
    for r in rows:
        print(f"{r[0]:10} {r[1]:5} {r[2]:6.3f} {r[3]:6.3f} "
              f"{r[4]:6.3f} {r[5]:6.3f}")

    print("\nHold-out ensemble F1:",
          f1_score(_y[ho], ens_ho, average="macro"))

    save_models([m for m in models if m[0] != "logreg"])

if __name__ == "__main__":
    main()
