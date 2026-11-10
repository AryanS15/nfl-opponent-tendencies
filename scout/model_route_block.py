"""
Predict ROUTE vs PASS-BLOCK for a skill player on a passing dropback,
using ONLY pre-snap information.

Guardrails (per task):
  - Inputs are pre-snap only: alignment (PFF snap label), official position, down,
    distance, formation, offensive personnel, field position, score diff, and a
    PLAYER HISTORICAL route rate computed from PAST GAMES ONLY (no leakage).
  - Excluded: pff_role (the label), post-snap tracking, outcomes, arrival events.
  - Temporal split: train on weeks 1-5, test on weeks 6-8. Every observation from a
    play stays in the same split (we split by week, and a play belongs to one week).
  - Baselines: (a) majority class, (b) position prior, (c) player historical route
    rate thresholded at 0.5.
  - Reports class balance, precision/recall/F1, ROC-AUC, calibration, and lift over
    baselines.

Scope statement: this predicts responsibility CONDITIONAL ON a passing dropback.
The dataset contains only dropbacks, so it CANNOT establish run-vs-pass tendencies.

Run: python3 scout/model_route_block.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             brier_score_loss, confusion_matrix, accuracy_score)
from sklearn.calibration import calibration_curve

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

TRAIN_WEEKS = {1, 2, 3, 4, 5}
TEST_WEEKS = {6, 7, 8}

def load():
    p = OUT / "player_play.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(OUT / "player_play.csv")

def prep(df):
    # model only players whose role is genuinely ambiguous: TE, RB, FB.
    # WR is ~99.8% route -> trivial; we report it but model TE/RB/FB.
    d = df[df.role_cat.isin(["route", "block"])].copy()
    d["y"] = (d.role_cat == "route").astype(int)
    d = d[d.week.notna()]
    return d

def player_hist_route_rate(train_df, full_df):
    """Historical route rate per player from PAST (training) data only.
    For each row we use the player's mean route rate over TRAIN weeks. Players unseen
    in train fall back to their position prior (also from train)."""
    player_rate = train_df.groupby("nflId")["y"].mean()
    pos_rate = train_df.groupby("officialPosition")["y"].mean()
    global_rate = train_df["y"].mean()
    def rate(row):
        if row.nflId in player_rate.index:
            return player_rate[row.nflId]
        if row.officialPosition in pos_rate.index:
            return pos_rate[row.officialPosition]
        return global_rate
    return full_df.apply(rate, axis=1), player_rate, pos_rate, global_rate

def main():
    df = prep(load())

    # Report the trivial WR case separately (not modeled)
    wr = load()
    wr = wr[wr.officialPosition == "WR"]
    wr_route = (wr.role_cat == "route").mean()
    print(f"WR route rate = {wr_route*100:.1f}%  (n={len(wr)}) -> trivial, excluded from model")

    # focus set: TE/RB/FB
    d = df[df.officialPosition.isin(["TE", "RB", "FB"])].copy()
    print(f"\nModeled observations (TE/RB/FB route-or-block): {len(d)}")
    print("class balance (route=1):")
    print(d["y"].value_counts(normalize=True).round(3).to_dict())

    train = d[d.week.isin(TRAIN_WEEKS)].copy()
    test = d[d.week.isin(TEST_WEEKS)].copy()
    print(f"train (wk1-5): {len(train)}   test (wk6-8): {len(test)}")
    print("train route rate:", round(train.y.mean(),3), " test route rate:", round(test.y.mean(),3))

    # historical feature from TRAIN ONLY, applied to both splits
    hr_all, player_rate, pos_rate, global_rate = player_hist_route_rate(train, d)
    d["hist_route_rate"] = hr_all
    train = d[d.week.isin(TRAIN_WEEKS)].copy()
    test = d[d.week.isin(TEST_WEEKS)].copy()

    # ----- features (ALL pre-snap) -----
    cat_feats = ["officialPosition", "align_bucket", "offenseFormation", "dd_bucket", "personnelO"]
    num_feats = ["down", "yardsToGo", "yards_to_goal", "hist_route_rate"]
    for c in cat_feats:
        train[c] = train[c].fillna("NA").astype(str)
        test[c] = test[c].fillna("NA").astype(str)
    for c in num_feats:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

    Xtr, ytr = train[cat_feats + num_feats], train.y.values
    Xte, yte = test[cat_feats + num_feats], test.y.values

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ("num", "passthrough", num_feats),
    ])

    models = {
        "logreg": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "gbm": Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=0))]),
    }

    # ----- baselines -----
    results = {}
    maj = int(round(ytr.mean()))
    base_majority = np.full(len(yte), maj)
    base_pos = test.officialPosition.map(pos_rate).fillna(global_rate).values
    base_hist = test.hist_route_rate.values

    def metrics(name, y_true, p_prob, p_pred=None, thr=0.5):
        if p_pred is None:
            p_pred = (p_prob >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, p_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, p_prob)
        except ValueError:
            auc = float("nan")
        brier = brier_score_loss(y_true, np.clip(p_prob,0,1))
        acc = accuracy_score(y_true, p_pred)
        return {"accuracy": round(acc,3), "precision": round(pr,3), "recall": round(rc,3),
                "f1": round(f1,3), "roc_auc": round(auc,3), "brier": round(brier,3)}

    results["baseline_majority"] = metrics("maj", yte, np.full(len(yte), ytr.mean()), base_majority)
    results["baseline_position_prior"] = metrics("pos", yte, base_pos)
    results["baseline_player_hist"] = metrics("hist", yte, base_hist)

    for name, m in models.items():
        m.fit(Xtr, ytr)
        prob = m.predict_proba(Xte)[:, 1]
        results[name] = metrics(name, yte, prob)
        if name == "gbm":
            # calibration curve
            frac_pos, mean_pred = calibration_curve(yte, prob, n_bins=10, strategy="quantile")
            results["gbm_calibration"] = {
                "mean_predicted": [round(float(x),3) for x in mean_pred],
                "fraction_positive": [round(float(x),3) for x in frac_pos],
            }
            cm = confusion_matrix(yte, (prob>=0.5).astype(int))
            results["gbm_confusion_matrix"] = {"tn":int(cm[0,0]),"fp":int(cm[0,1]),
                                               "fn":int(cm[1,0]),"tp":int(cm[1,1])}
            # persist model
            import joblib
            joblib.dump(m, OUT / "route_block_gbm.joblib")

    # lift over best baseline (by F1 and AUC)
    print("\n" + "="*70 + "\nRESULTS (test = weeks 6-8)\n" + "="*70)
    order = ["baseline_majority","baseline_position_prior","baseline_player_hist","logreg","gbm"]
    hdr = f"{'model':26s} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'auc':>6} {'brier':>6}"
    print(hdr); print("-"*len(hdr))
    for k in order:
        r = results[k]
        print(f"{k:26s} {r['accuracy']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['f1']:6.3f} {r['roc_auc']:6.3f} {r['brier']:6.3f}")

    best_base_auc = max(results["baseline_position_prior"]["roc_auc"],
                        results["baseline_player_hist"]["roc_auc"])
    print(f"\nGBM AUC lift over best baseline: "
          f"{results['gbm']['roc_auc'] - best_base_auc:+.3f}")
    print("\nCalibration (gbm):")
    for mp, fp in zip(results["gbm_calibration"]["mean_predicted"],
                      results["gbm_calibration"]["fraction_positive"]):
        print(f"   predicted ~{mp:.2f} -> actual {fp:.2f}")
    print("\nConfusion matrix (gbm):", results["gbm_confusion_matrix"])

    (OUT / "model_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "model_results.json")
    print("wrote", OUT / "route_block_gbm.joblib")

    # persist train-period historical rates so the dashboard can score held-out
    # (weeks 6-8) observations with leakage-free features.
    import joblib as _jl
    _jl.dump({
        "player_rate": player_rate,
        "pos_rate": pos_rate,
        "global_rate": float(global_rate),
        "train_weeks": sorted(TRAIN_WEEKS),
        "test_weeks": sorted(TEST_WEEKS),
    }, OUT / "hist_rates.joblib")
    print("wrote", OUT / "hist_rates.joblib")

    print("\nSCOPE: predicts route-vs-block CONDITIONAL on a passing dropback. "
          "Dataset has only dropbacks; it cannot model run-vs-pass.")

if __name__ == "__main__":
    main()
