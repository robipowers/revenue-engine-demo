"""Propensity model: seeded logistic regression on won/lost opportunities.

Training computes every feature AS OF each opportunity's created_at (see
features.py for the leakage guard) and reports AUC on a stratified holdout.
`score` is a pure function: pass the trained model in (or load it once);
same input always yields the same output, no I/O.

CLI:

    python -m revenue_engine.model --train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from revenue_engine import warehouse
from revenue_engine.config import MODELS_DIR, RANDOM_SEED, SEED_DB
from revenue_engine.features import FEATURE_NAMES, features_as_of, vectorize

MODEL_PATH = MODELS_DIR / "propensity.joblib"


def build_training_data(con) -> tuple[list[list[float]], list[int]]:
    """One row per closed (won/lost) opportunity, features as of created_at."""
    opps = con.execute(
        "SELECT id, account_id, outcome, created_at FROM opportunities "
        "WHERE outcome IN ('won', 'lost') ORDER BY id"
    ).fetchall()
    X, y = [], []
    for _opp_id, account_id, outcome, created_at in opps:
        X.append(vectorize(features_as_of(con, account_id, created_at)))
        y.append(1 if outcome == "won" else 0)
    return X, y


def train(
    db_path: Path | str = SEED_DB,
    model_path: Path | str = MODEL_PATH,
    test_size: float = 0.3,
) -> dict:
    """Train, evaluate on a stratified holdout, persist, return the payload."""
    con = warehouse.connect(db_path)
    try:
        X, y = build_training_data(con)
    finally:
        con.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    auc = float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))

    payload = {
        "model": pipeline,
        "feature_names": list(FEATURE_NAMES),
        "holdout_auc": auc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_positive": int(sum(y)),
        "n_total": len(y),
        "random_seed": RANDOM_SEED,
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    return payload


def load(model_path: Path | str = MODEL_PATH) -> dict:
    """Load a persisted payload (call once; then score() is pure)."""
    return joblib.load(model_path)


def score(features: dict, model) -> float:
    """Win-propensity in [0, 1] for one feature dict. Pure: no I/O.

    `model` is either the payload dict from train()/load() or the sklearn
    pipeline itself.
    """
    pipeline = model["model"] if isinstance(model, dict) else model
    vector = vectorize(features)
    return float(pipeline.predict_proba([vector])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Propensity model management")
    parser.add_argument("--train", action="store_true", help="train and persist the model")
    args = parser.parse_args()
    if args.train:
        payload = train()
        print(f"trained on {payload['n_train']} rows, holdout {payload['n_test']} rows")
        print(f"class balance: {payload['n_positive']}/{payload['n_total']} won")
        print(f"holdout AUC: {payload['holdout_auc']:.3f}")
        print(f"saved {MODEL_PATH}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
