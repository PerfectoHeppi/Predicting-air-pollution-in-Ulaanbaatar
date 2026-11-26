"""Summarize matched holdout and out-of-fold predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import CANDIDATES, CURRENT, ROOT, add_blends, split_masks


def load_scores(directory, week, actual, include_embassy):
    results = []
    stages = ["current", "features", "catboost"]
    if include_embassy:
        stages.append("embassy")
    for fold in range(6):
        _, valid = split_masks(week, fold)
        joined = pd.DataFrame({"actual": actual.loc[valid]})
        for stage in stages:
            frame = pd.read_csv(directory / f"{stage}-{fold}.csv", index_col=0)
            assert frame.index.equals(joined.index)
            if "actual" in frame:
                np.testing.assert_allclose(frame.pop("actual"), joined.actual)
            joined = joined.join(frame)
        assert np.isfinite(joined.to_numpy()).all()
        results.append(add_blends(joined))
    all_validation = pd.concat(results[1:], verify_integrity=True).sort_index()
    assert all_validation.index.equals(actual.index)
    return results, all_validation


def bootstrap_delta(squared, week, name, samples):
    by_week = squared[[CURRENT, name]].groupby(week).sum().to_numpy()
    counts = week.value_counts().sort_index().to_numpy()
    errors = by_week[samples].sum(axis=1)
    rmse = np.sqrt(errors / counts[samples].sum(axis=1)[:, None])
    return np.quantile(rmse[:, 1] - rmse[:, 0], (0.025, 0.975))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--skip-embassy", action="store_true")
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "Data/train.csv", parse_dates=["datetime"])
    train = train.sort_values(["station", "datetime"]).reset_index(drop=True)
    week = train.datetime.dt.to_period("W")
    results, all_validation = load_scores(args.directory, week, train.pm25, not args.skip_embassy)
    squared = [frame.drop(columns="actual").sub(frame.actual, axis=0).pow(2) for frame in [*results, all_validation]]
    columns = ["Holdout", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "OOF"]
    scores = pd.DataFrame(dict(zip(columns, [np.sqrt(frame.mean()) for frame in squared], strict=True)))
    scores["Mean CV"] = scores[columns[1:6]].mean(axis=1)
    scores["CV SD"] = scores[columns[1:6]].std(axis=1, ddof=0)
    scores["OOF delta"] = scores.OOF - scores.loc[CURRENT, "OOF"]
    print(scores.round(4).to_string())
    samples = np.random.default_rng(42).integers(week.nunique(), size=(4000, week.nunique()))
    print("\nCandidate − current peak. Negative favors the candidate.")
    for name in CANDIDATES:
        if name not in scores.index:
            continue
        lo, hi = bootstrap_delta(squared[-1], week, name, samples)
        better = (scores.loc[name, columns[1:6]] < scores.loc[CURRENT, columns[1:6]]).sum()
        holdout = scores.loc[name, "Holdout"] - scores.loc[CURRENT, "Holdout"]
        print(f"{name}: 95% week bootstrap [{lo:+.4f}, {hi:+.4f}], "
              f"holdout {holdout:+.4f}, folds better {better}/5")


if __name__ == "__main__":
    main()
