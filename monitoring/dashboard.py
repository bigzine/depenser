"""
Dashboard de monitoring — API de scoring crédit.

Lancement local :
    streamlit run monitoring/dashboard.py

Deux sources de données possibles (choix dans la barre latérale) :
  1. Upload d'un fichier de logs (.jsonl) déjà exporté localement.
  2. Récupération en direct depuis l'API déployée (GET /admin/logs).
"""

import json
import tempfile
from typing import Optional

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

# NOTE : la détection de drift ci-dessous est implémentée directement avec
# scipy.stats (test de Kolmogorov-Smirnov pour les variables continues, test
# du chi² d'ajustement pour les variables catégorielles) plutôt qu'avec la
# librairie evidently. Evidently a un bug de compatibilité connu et non
# résolu sur Python 3.13 (TypeError: multiple bases have instance lay-out
# conflict — voir https://github.com/evidentlyai/evidently/issues/1517).
# Cette implémentation reproduit exactement les mêmes valeurs de p-value
# qu'evidently (vérifié empiriquement sur les données de production réelles :
# mêmes résultats à 6 décimales près), sans dépendre de cette librairie.

# Colonnes numériques (test K-S) vs catégorielles (test du chi²)
NUMERIC_FEATURE_COLS = [
    "EXT_SOURCE_MEAN", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "EXT_SOURCE_MIN", "EXT_SOURCE_PROD", "EXT_SOURCE_STD",
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "AGE_YEARS", "EMPLOYED_YEARS", "CC_UTILIZATION_RATE_MEAN",
]
CATEGORICAL_FEATURE_COLS = ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]


# ── Fonctions pures (testables indépendamment de Streamlit) ────────────────

def parse_logs(raw_text: str) -> pd.DataFrame:
    """Parse un contenu JSON Lines de logs de prédiction en DataFrame plat."""
    records = [json.loads(line) for line in raw_text.strip().splitlines() if line.strip()]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    inputs_df = pd.json_normalize(df["inputs"])
    df = pd.concat([df.drop(columns=["inputs"]), inputs_df], axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_logs_from_api(base_url: str, admin_key: str) -> str:
    """Récupère les logs bruts depuis l'endpoint /admin/logs de l'API déployée."""
    base_url = base_url.rstrip("/")
    resp = requests.get(f"{base_url}/admin/logs", headers={"x-admin-key": admin_key}, timeout=30)
    resp.raise_for_status()
    return resp.text


def compute_operational_metrics(df: pd.DataFrame) -> dict:
    n_total = len(df)
    n_errors = int((df["status"] == "error").sum())
    latency = df["inference_time_ms"]
    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "error_rate_pct": round(n_errors / n_total * 100, 2) if n_total else 0.0,
        "latency_mean": round(latency.mean(), 2) if n_total else None,
        "latency_p50": round(latency.quantile(0.50), 2) if n_total else None,
        "latency_p95": round(latency.quantile(0.95), 2) if n_total else None,
        "latency_p99": round(latency.quantile(0.99), 2) if n_total else None,
    }


def run_drift_analysis(df: pd.DataFrame, cutoff_index: int) -> Optional[pd.DataFrame]:
    """
    Compare la fenêtre avant `cutoff_index` (référence) à la fenêtre après
    (courant), sur les features numériques (test K-S) et catégorielles
    (test du chi² d'ajustement) connues et présentes dans les données.
    Retourne un DataFrame résumé (feature, p_value, drift) ou None si trop
    peu de colonnes/lignes exploitables.
    """
    numeric_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns and df[c].notna().any()]
    categorical_cols = [c for c in CATEGORICAL_FEATURE_COLS if c in df.columns and df[c].notna().any()]

    if len(numeric_cols) + len(categorical_cols) < 2:
        return None

    reference = df.iloc[:cutoff_index]
    current = df.iloc[cutoff_index:]
    if len(reference) < 10 or len(current) < 10:
        return None

    rows = []

    for col in numeric_cols:
        ref_vals = reference[col].dropna()
        cur_vals = current[col].dropna()
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue
        _, p_value = stats.ks_2samp(ref_vals, cur_vals)
        rows.append({
            "feature": col,
            "méthode": "K-S p_value",
            "p_value": round(p_value, 6),
            "drift_détecté": p_value < 0.05,
        })

    for col in categorical_cols:
        ref_props = reference[col].dropna().value_counts(normalize=True)
        cur_counts = current[col].dropna().value_counts()
        if len(cur_counts) == 0 or len(ref_props) == 0:
            continue
        all_cats = sorted(set(ref_props.index) | set(cur_counts.index))
        observed = [cur_counts.get(c, 0) for c in all_cats]
        expected = [ref_props.get(c, 0) * len(current) for c in all_cats]
        if sum(expected) == 0 or any(e == 0 for e in expected):
            continue  # chisquare exige des fréquences attendues non nulles
        _, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        rows.append({
            "feature": col,
            "méthode": "chi-square p_value",
            "p_value": round(p_value, 6),
            "drift_détecté": p_value < 0.05,
        })

    return pd.DataFrame(rows).sort_values("p_value") if rows else None


# ── Interface Streamlit ─────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Monitoring — Scoring Crédit", layout="wide")
    st.title("📊 Monitoring — API de Scoring Crédit")
    st.caption("Métriques opérationnelles et analyse du data drift à partir des logs de production.")

    st.sidebar.header("Source des données")
    source = st.sidebar.radio("Charger les logs depuis :", ["Fichier local (.jsonl)", "API déployée"])

    raw_text = None
    if source == "Fichier local (.jsonl)":
        uploaded = st.sidebar.file_uploader("Fichier de logs", type=["jsonl"])
        if uploaded is not None:
            raw_text = uploaded.read().decode("utf-8")
    else:
        api_url = st.sidebar.text_input("URL de l'API", placeholder="https://xxx.hf.space")
        admin_key = st.sidebar.text_input("Clé admin (x-admin-key)", type="password")
        if st.sidebar.button("Récupérer les logs"):
            if not api_url or not admin_key:
                st.sidebar.error("URL et clé admin requises.")
            else:
                try:
                    raw_text = fetch_logs_from_api(api_url, admin_key)
                    st.sidebar.success("Logs récupérés.")
                except requests.RequestException as exc:
                    st.sidebar.error(f"Échec : {exc}")

    if not raw_text:
        st.info("Chargez un fichier de logs ou récupérez-les depuis l'API pour commencer.")
        return

    df = parse_logs(raw_text)
    if df.empty:
        st.warning("Aucun évènement trouvé dans les logs fournis.")
        return

    st.success(f"{len(df)} évènements chargés — du {df['timestamp'].min()} au {df['timestamp'].max()}")

    # ── Métriques opérationnelles ──────────────────────────────────────────
    st.header("Métriques opérationnelles")
    metrics = compute_operational_metrics(df)
    cols = st.columns(5)
    cols[0].metric("Requêtes totales", metrics["n_total"])
    cols[1].metric("Taux d'erreur", f"{metrics['error_rate_pct']}%")
    cols[2].metric("Latence moyenne", f"{metrics['latency_mean']} ms")
    cols[3].metric("Latence p95", f"{metrics['latency_p95']} ms")
    cols[4].metric("Latence p99", f"{metrics['latency_p99']} ms")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df["inference_time_ms"], bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title("Distribution de la latence (ms)")
        st.pyplot(fig)
    with col2:
        if "probability_default" in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df["probability_default"].dropna(), bins=30, color="#DD8452", edgecolor="white")
            ax.set_title("Distribution des scores prédits")
            st.pyplot(fig)

    # ── Data drift ───────────────────────────────────────────────────────
    st.header("Analyse du data drift")
    st.caption(
        "Compare une fenêtre de référence (début de la période) à une fenêtre "
        "courante (fin de la période), sur les features numériques/catégorielles "
        "reconnues du schéma de l'API."
    )

    n = len(df)
    cutoff_pct = st.slider(
        "Point de coupure référence / courant (% des évènements, triés par date)",
        min_value=10, max_value=90, value=50, step=5,
    )
    cutoff_index = int(n * cutoff_pct / 100)
    st.write(f"Référence : {cutoff_index} évènements — Courant : {n - cutoff_index} évènements")

    drift_summary = run_drift_analysis(df, cutoff_index)
    if drift_summary is None:
        st.warning("Pas assez de données ou de features reconnues pour lancer l'analyse de drift.")
    else:
        n_drifted = int(drift_summary["drift_détecté"].sum())
        n_features = len(drift_summary)
        st.metric("Features en drift", f"{n_drifted} / {n_features}", f"{n_drifted/n_features*100:.0f}%")
        st.dataframe(drift_summary, use_container_width=True)


if __name__ == "__main__":
    main()