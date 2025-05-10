from pathlib import Path
import json, re, joblib, warnings, pandas as pd, numpy as np, streamlit as st
from catboost import CatBoostClassifier
from xgboost   import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

MODEL_DIR = Path("models")

def latest_manifest() -> Path:
    manifests = sorted(MODEL_DIR.glob("ensemble_*.json"))
    if not manifests:
        st.error("No trained models were found in the ./models directory.")
        st.stop()
    return manifests[-1]

def _patch_rf_monotonic(rf: RandomForestClassifier):

    for tree in rf.estimators_:
        if not hasattr(tree, "monotonic_cst"):
            tree.monotonic_cst = None

def _load_models_uncached():
    meta   = json.load(open(latest_manifest()))
    models = []
    for fname, info in meta.items():
        mtype = info["type"] if isinstance(info, dict) else info[0]
        wt    = info["weight"] if isinstance(info, dict) else info[1]
        p     = MODEL_DIR / fname

        if mtype == "catboost":
            mdl = CatBoostClassifier(); mdl.load_model(p)
        else:
            mdl = joblib.load(p)
            if isinstance(mdl, RandomForestClassifier):
                _patch_rf_monotonic(mdl)

        models.append((fname.split("_")[0], mdl, wt))
    return models

@st.cache_resource
def load_models_cached():
    return _load_models_uncached()
def encode_for_rf(df: pd.DataFrame) -> pd.DataFrame:
    df_rf = df.copy()
    for c in df_rf.select_dtypes("category"):
        df_rf[c] = df_rf[c].cat.codes.astype("int32")
    df_rf = df_rf.drop(columns=["Country", "state"], errors="ignore")
    return df_rf

def build_full_row(user_inputs: dict, models):

    cb_model   = next(m for m in models if m[0].startswith("cb"))[1]
    feat_names = cb_model.feature_names_
    cat_idx    = set(cb_model.get_cat_feature_indices())

    full = pd.DataFrame([{c: user_inputs.get(c, np.nan) for c in feat_names}])

    for i, col in enumerate(feat_names):
        if i in cat_idx:
            full[col] = (full[col].fillna("Unknown")
                                   .astype("string[pyarrow]")
                                   .astype("category"))
        else:
            full[col] = pd.to_numeric(full[col], errors="coerce").fillna(-1)

    df_rf = encode_for_rf(full)
    return full, df_rf


def _safe_proba(mdl, X):

    p = mdl.predict_proba(X).astype(np.float64)
    s = p.sum(axis=1, keepdims=True)
    return np.divide(p, s, where=s != 0)

def predict_ensemble(models, df_cb, df_rf):
    weights = np.array([w for *_, w in models], dtype=np.float64)
    proba   = np.zeros((len(df_cb), 3), dtype=np.float64)

    for name, mdl, w in models:
        if name.startswith("cb") or name == "xgb":
            proba += _safe_proba(mdl, df_cb) * w
        elif name == "rf":
            proba += _safe_proba(mdl, df_rf) * w

    proba /= weights.sum()
    return proba, proba.argmax(axis=1)


st.set_page_config(page_title="OSMI Work-Interference Predictor",
                   layout="centered")

st.title("Mental-health Work-Interference Prediction")

with st.sidebar:
    st.header("Fill in your answers")

    ui = st.session_state.setdefault("ui", {})
    def sel(label, opts): return st.selectbox(label, opts, key=label, index=0)

    ui["Age"]  = st.number_input("Age",   18, 100, value=ui.get("Age", 30))
    ui["Gender"]       = sel("Gender", ["Male", "Female", "Other"])
    ui["Country"]      = st.text_input("Country", ui.get("Country", "United States"))
    ui["self_employed"]= sel("Self-employed", ["No", "Yes"])
    ui["family_history"]= sel("Family history of mental illness", ["No", "Yes"])
    ui["treatment"]    = sel("Have you sought treatment?", ["No", "Yes"])
    ui["no_employees"] = sel("Company size",
                             ["1-5","6-25","26-100","100-500","500-1000","More than 1000"])
    ui["remote_work"]  = sel("Work remotely ≥50 %?", ["No", "Yes"])
    ui["tech_company"] = sel("Employer is primarily tech?", ["No", "Yes"])
    ui["benefits"]     = sel("Mental-health benefits provided?", ["Don't know","No","Yes"])
    ui["care_options"] = sel("Aware of care options?", ["Not sure","No","Yes"])
    ui["wellness_program"] = sel("Employer discussed MH in wellness program?",
                                 ["Don't know","No","Yes"])
    ui["seek_help"]    = sel("Employer provides MH resources?", ["Don't know","No","Yes"])
    ui["anonymity"]    = sel("Anonymity protected?", ["Don't know","No","Yes"])
    ui["leave"]        = sel("Ease of MH medical leave",
        ["Very difficult","Somewhat difficult","Somewhat easy","Very easy","Don't know"])
    ui["mental_health_consequence"] = sel("Negative consequence if you discuss MH?",
                                          ["Maybe","No","Yes"])
    ui["phys_health_consequence"]   = sel("Negative consequence if you discuss PH?",
                                          ["Maybe","No","Yes"])
    ui["coworkers"]   = sel("Discuss MH with coworkers?", ["No","Some of them","Yes"])
    ui["supervisor"]  = sel("Discuss MH with supervisor?", ["No","Some of them","Yes"])
    ui["mental_health_interview"] = sel("Bring up MH in interview?", ["No","Maybe","Yes"])
    ui["phys_health_interview"]   = sel("Bring up PH in interview?", ["No","Maybe","Yes"])
    ui["mental_vs_physical"] = sel("Employer treats MH same as physical?",
                                   ["Don't know","No","Yes"])
    ui["obs_consequence"]  = sel("Observed negative MH consequences at work?", ["No","Yes"])

    if st.button("Predict"):
        st.session_state["go"] = True

if st.session_state.get("go", False):
    with st.spinner("Loading ensemble and making prediction…"):
        models       = load_models_cached()
        df_cb, df_rf = build_full_row(ui, models)
        proba, pred  = predict_ensemble(models, df_cb, df_rf)

    label_map = {0: "No / Rarely interferes",
                 1: "Sometimes interferes",
                 2: "Often interferes"}

    st.subheader("Predicted interference level")
    st.metric("Prediction", label_map[int(pred[0])])
    st.caption(
        f"Class probabilities → "
        f"0: {proba[0,0]:.2f} • 1: {proba[0,1]:.2f} • 2: {proba[0,2]:.2f}"
    )

    # per-model breakdown
    with st.expander("Individual model outputs", expanded=False):
        for name, mdl, _ in models:
            pp = _safe_proba(mdl, df_rf if name == "rf" else df_cb)[0]
            st.write(
                f"**{name.upper()}** → {label_map[int(pp.argmax())]} "
                f"(probs: {pp[0]:.2f}, {pp[1]:.2f}, {pp[2]:.2f})"
            )

    # global feature importances
    with st.expander("Top global feature importances", expanded=False):
        for name, mdl, _ in models:
            if name.startswith("cb"):
                fi = mdl.get_feature_importance(type="FeatureImportance")
                idx = np.argsort(fi)[::-1][:10]
                st.write(f"**{name.upper()}**")
                st.table(pd.DataFrame({
                    "feature": [df_cb.columns[i] for i in idx],
                    "importance": fi[idx].round(2)
                }))
            elif name == "xgb":
                fi = mdl.get_booster().get_score(importance_type="gain")
                top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
                st.write("**XGB**")
                st.table(pd.DataFrame(top, columns=["feature", "gain"]).round(2))
            elif name == "rf":
                fi  = mdl.feature_importances_
                idx = np.argsort(fi)[::-1][:10]
                st.write("**RF**")
                st.table(pd.DataFrame({
                    "feature": [df_rf.columns[i] for i in idx],
                    "importance": fi[idx].round(2)
                }))

    # reset flag so next click re-runs prediction
    st.session_state["go"] = False
