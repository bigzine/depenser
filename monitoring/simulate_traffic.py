"""
Simulateur de trafic pour l'API de scoring déployée.

Génère deux populations de requêtes :
  1. "Normale" — features tirées de distributions proches des statistiques
     réelles observées sur le jeu d'entraînement (cf. notebook étape 2).
  2. "Dérivée" (drift) — mêmes features mais avec des distributions
     volontairement décalées, pour simuler un changement de population
     de clients dans le temps (data drift).

Usage :
    python monitoring/simulate_traffic.py --url https://ediagabate-scoring-credit-api.hf.space \
        --n-normal 150 --n-drifted 150

Les requêtes sont journalisées côté API (logs/predictions.jsonl) ; ce script
ne fait qu'appeler /predict de façon répétée, il n'écrit rien lui-même.
"""

import argparse
import concurrent.futures
import sys
import time

import numpy as np
import requests

# ── Statistiques de référence (issues du notebook étape 2, jeu d'entraînement) ─
# EXT_SOURCE_* : moyenne/écart-type réels observés sur X_train.
REF_STATS = {
    "EXT_SOURCE_MEAN": (0.5093, 0.1498),
    "EXT_SOURCE_MIN": (0.3996, 0.1874),
    "EXT_SOURCE_2": (0.5145, 0.1909),
    "EXT_SOURCE_3": (0.5157, 0.1747),
    "EXT_SOURCE_PROD": (0.2502, 0.1803),
}

# AMT_CREDIT, AMT_ANNUITY, AGE_YEARS : valeurs approximatives basées sur les
# statistiques publiques connues du jeu de données Home Credit Default Risk
# (à ajuster si vous disposez des statistiques exactes de votre X_train).
APPROX_STATS = {
    "AMT_CREDIT": (599_026, 402_490),
    "AMT_ANNUITY": (27_109, 14_494),
    "AGE_YEARS": (43.0, 11.5),
}

REGION_RATING_WEIGHTS = {1: 0.10, 2: 0.70, 3: 0.20}


def sample_normal_client(rng: np.random.Generator) -> dict:
    """Un client 'normal', proche de la distribution d'entraînement."""
    ext_mean, ext_std = REF_STATS["EXT_SOURCE_MEAN"]
    ext2_mean, ext2_std = REF_STATS["EXT_SOURCE_2"]
    ext3_mean, ext3_std = REF_STATS["EXT_SOURCE_3"]
    credit_mean, credit_std = APPROX_STATS["AMT_CREDIT"]
    annuity_mean, annuity_std = APPROX_STATS["AMT_ANNUITY"]
    age_mean, age_std = APPROX_STATS["AGE_YEARS"]

    return {
        "EXT_SOURCE_MEAN": float(np.clip(rng.normal(ext_mean, ext_std), 0, 1)),
        "EXT_SOURCE_2": float(np.clip(rng.normal(ext2_mean, ext2_std), 0, 1)),
        "EXT_SOURCE_3": float(np.clip(rng.normal(ext3_mean, ext3_std), 0, 1)),
        "AMT_CREDIT": float(max(45_000, rng.normal(credit_mean, credit_std))),
        "AMT_ANNUITY": float(max(1_500, rng.normal(annuity_mean, annuity_std))),
        "AGE_YEARS": float(np.clip(rng.normal(age_mean, age_std), 18, 100)),
        "REGION_RATING_CLIENT": int(
            rng.choice(list(REGION_RATING_WEIGHTS), p=list(REGION_RATING_WEIGHTS.values()))
        ),
    }


def sample_drifted_client(rng: np.random.Generator) -> dict:
    """
    Un client 'dérivé' : distributions volontairement décalées pour simuler
    un data drift réaliste — population plus risquée (scores externes plus
    bas) demandant des crédits plus élevés.
    """
    base = sample_normal_client(rng)
    base["EXT_SOURCE_MEAN"] = float(np.clip(base["EXT_SOURCE_MEAN"] - 0.20, 0, 1))
    base["EXT_SOURCE_2"] = float(np.clip(base["EXT_SOURCE_2"] - 0.20, 0, 1))
    base["EXT_SOURCE_3"] = float(np.clip(base["EXT_SOURCE_3"] - 0.20, 0, 1))
    base["AMT_CREDIT"] = float(base["AMT_CREDIT"] * 1.6)
    base["AGE_YEARS"] = float(np.clip(base["AGE_YEARS"] - 12, 18, 100))
    return base


def send_request(url: str, sk_id_curr: int, features: dict) -> dict:
    payload = {"sk_id_curr": sk_id_curr, "features": features}
    try:
        r = requests.post(f"{url}/predict", json=payload, timeout=15)
        return {"sk_id_curr": sk_id_curr, "status_code": r.status_code}
    except requests.RequestException as exc:
        return {"sk_id_curr": sk_id_curr, "status_code": None, "error": str(exc)}


def run_batch(url: str, n: int, sampler, start_id: int, label: str, max_workers: int = 8):
    rng = np.random.default_rng()
    print(f"→ Envoi de {n} requêtes '{label}'...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(send_request, url, start_id + i, sampler(rng))
            for i in range(n)
        ]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if i % 25 == 0 or i == n:
                print(f"  {i}/{n} envoyées")

    n_ok = sum(1 for r in results if r["status_code"] == 200)
    n_fail = n - n_ok
    print(f"✅ Batch '{label}' terminé : {n_ok} succès, {n_fail} échecs\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Simule du trafic vers l'API de scoring déployée.")
    parser.add_argument("--url", required=True, help="URL de base de l'API (sans slash final)")
    parser.add_argument("--n-normal", type=int, default=150)
    parser.add_argument("--n-drifted", type=int, default=150)
    args = parser.parse_args()

    url = args.url.rstrip("/")

    # Vérification rapide que l'API est joignable avant de lancer le batch.
    try:
        health = requests.get(f"{url}/health", timeout=10)
        health.raise_for_status()
        print(f"✅ API joignable : {health.json()}\n")
    except requests.RequestException as exc:
        print(f"❌ Impossible de joindre l'API à {url} : {exc}")
        sys.exit(1)

    t0 = time.time()
    run_batch(url, args.n_normal, sample_normal_client, start_id=1, label="normale")
    run_batch(url, args.n_drifted, sample_drifted_client, start_id=100_000, label="dérivée (drift)")
    elapsed = time.time() - t0

    print(f"Terminé en {elapsed:.1f}s. Récupérez les logs via :")
    print(f'  curl {url}/admin/logs -H "x-admin-key: <votre_clé>" > monitoring/production_logs.jsonl')


if __name__ == "__main__":
    main()