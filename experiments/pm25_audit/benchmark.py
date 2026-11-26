"""Compare PM2.5 models on identical, week-grouped validation sets."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import peak_reference as app

BASE_COLS = ["station", "hour", "dayofweek", "month", "temperature", "humidity"]
BASELINE_PARAMS = {
    **app.MODEL_PARAMS, "learning_rate": 0.005, "num_leaves": 10,
    "min_data_in_leaf": 120, "feature_fraction": 0.8, "bagging_fraction": 0.8,
}
STANDARD_PARAMS = {**app.MODEL_PARAMS, "learning_rate": 0.065, "max_bin": 95}
TRIAL_PARAMS = [
    STANDARD_PARAMS,
    {**STANDARD_PARAMS, "learning_rate": 0.085, "num_leaves": 95,
     "min_data_in_leaf": 55, "lambda_l2": 4.0, "max_bin": 63},
]
ROUND_OPTIONS = (200, 350, 500)
CAT_ROUND_OPTIONS = (250, 450, 650)
FORECAST_BLEND_WEIGHTS = (0.50, 0.40, 0.10)
CURRENT = "Current peak blend"
CANDIDATES = (
    "Weather features", "Tuned LightGBM", "CatBoost",
    "Peak + tuned + CatBoost", "Peak + tuned", "Embassy lags", "Peak + Embassy lags",
)
EMBASSY_COLUMNS = ["Date (LT)", "Parameter", "QC Name", "Raw Conc."]


def load_inputs(weather_json):
    original = pd.read_csv(ROOT / "Data/train.csv", parse_dates=["datetime"])
    train = original.sort_values(["station", "datetime"]).reset_index(drop=True)
    test = pd.read_csv(ROOT / "Data/test.csv", parse_dates=["datetime"])
    first = min(train.datetime.min(), test.datetime.min())
    last = max(train.datetime.max(), test.datetime.max())
    weather = app.load_weather(first, last, weather_json)
    categories = sorted(train.station.unique())
    basic, basic_test = app.make_features(train, test, categories, weather)
    peak, peak_test = app.make_features(train, test, categories, weather.join(app.peak_weather_features(weather)))
    enhanced, enhanced_test = add_weather_features(train, test, peak, peak_test, weather)
    week = train.datetime.dt.to_period("W")
    return original, train, test, basic, basic_test, peak, peak_test, enhanced, enhanced_test, week


def add_weather_features(train, test, peak, peak_test, weather):
    extra = pd.DataFrame(index=weather.index)
    for horizon in (3, 12, 48):
        extra[f"ventilation_min_{horizon}h"] = weather.era5_ventilation.rolling(horizon).min()
        extra[f"temperature_mean_{horizon}h"] = weather.era5_temperature_2m.rolling(horizon).mean()
    direction_x = weather.era5_wind_speed_10m * weather.era5_wind_direction_cos
    direction_y = weather.era5_wind_speed_10m * weather.era5_wind_direction_sin
    for axis, direction in (("x", direction_x), ("y", direction_y)):
        extra[f"wind_{axis}"] = direction
        extra[f"wind_{axis}_mean_12h"] = direction.rolling(12).mean()
    extra["wind_shear"] = weather.era5_wind_speed_100m - weather.era5_wind_speed_10m

    def append(rows, features):
        result = features.join(extra.reindex(rows.datetime).reset_index(drop=True))
        result["station_minus_era5_temperature"] = result.temperature - result.era5_temperature_2m
        result["station_minus_era5_humidity"] = result.humidity - result.era5_relative_humidity_2m
        result["temperature_missing"] = result.temperature.isna().astype("int8")
        return result

    return append(train, peak), append(test, peak_test)


def load_embassy(directory):
    files = sorted(directory.glob("*_YTD.csv"))
    if not files:
        raise ValueError(f"No Embassy CSV files found in {directory}")
    rows = pd.concat([pd.read_csv(file, usecols=EMBASSY_COLUMNS) for file in files], ignore_index=True)
    valid = rows["Parameter"].eq("PM2.5 - Principal") & rows["QC Name"].eq("Valid") & rows["Raw Conc."].ge(0)
    measured = rows.loc[valid].assign(time=lambda frame: pd.to_datetime(frame["Date (LT)"], format="%Y-%m-%d %I:%M %p"))
    concentration = measured.groupby("time")["Raw Conc."].mean()
    hourly = concentration.reindex(pd.date_range(concentration.index.min(), concentration.index.max(), freq="h"))
    # Explicit two-hour availability assumption for the historical lag experiment.
    lagged = hourly.shift(2)
    past = pd.DataFrame({f"emb_lag_{lag}h": hourly.shift(lag) for lag in (2, 3, 4, 6, 12, 24)})
    past["emb_mean_asof_12h"] = lagged.rolling(12, min_periods=1).mean()
    past["emb_mean_asof_24h"] = lagged.rolling(24, min_periods=1).mean()
    past["emb_max_asof_24h"] = lagged.rolling(24, min_periods=1).max()
    legacy = pd.DataFrame({f"emb_t{k:+d}": hourly.shift(-k) for k in (-1, 0, 1, 2, 3, 4)})
    legacy["emb_mean_t0_t4"] = hourly.shift(-4).rolling(5, min_periods=1).mean()
    legacy["emb_mean_12h"] = hourly.rolling(12, center=True, min_periods=1).mean()
    legacy["emb_mean_24h"] = hourly.rolling(24, center=True, min_periods=1).mean()
    legacy["emb_max_24h"] = hourly.rolling(24, center=True, min_periods=1).max()
    return past, legacy


def split_masks(week, fold):
    groups = np.sort(week.unique())
    if fold == 0:
        groups = np.random.default_rng(0).choice(week.unique(), int(len(groups) * 0.3), replace=False)
    else:
        groups = np.array_split(np.random.default_rng(42).permutation(groups), 5)[fold - 1]
    valid = week.isin(groups)
    assert set(week.loc[~valid]).isdisjoint(week.loc[valid])
    return ~valid, valid


def inner_validation(week):
    groups = np.sort(week.unique())
    validation_weeks = np.random.default_rng(17).choice(groups, size=len(groups) // 5, replace=False)
    return week.isin(validation_weeks)


def lightgbm_fit(X, y, params, rounds):
    return app.lgb.train(params, app.lgb.Dataset(X, label=y), num_boost_round=rounds)


def predict_lgb(model, X, rounds=None):
    return model.predict(X, num_iteration=rounds, num_threads=1).clip(min=0)


def select_lgb(X, y, week):
    val = inner_validation(week)
    trials = []
    for config, params in enumerate(TRIAL_PARAMS):
        model = lightgbm_fit(X.loc[~val], y.loc[~val], params, max(ROUND_OPTIONS))
        for rounds in ROUND_OPTIONS:
            score = app.rmse(y.loc[val], predict_lgb(model, X.loc[val], rounds))
            trials.append((score, config, rounds))
    return min(trials)


def cat_frame(X):
    return X.assign(station=X.station.astype(str))


def catboost_fit(X, y, rounds):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=rounds, learning_rate=0.085, depth=7, cat_features=["station"],
        loss_function="RMSE", random_seed=42, thread_count=1, allow_writing_files=False,
        verbose=False,
    ).fit(X, y)


def select_catboost(X, y, week):
    val = inner_validation(week)
    model = catboost_fit(X.loc[~val], y.loc[~val], max(CAT_ROUND_OPTIONS))
    return min(
        (app.rmse(y.loc[val], model.predict(X.loc[val], ntree_end=rounds).clip(min=0)), rounds)
        for rounds in CAT_ROUND_OPTIONS
    )


def benchmark(args, inputs):
    _, train, _, basic, _, peak, _, enhanced, _, week = inputs
    y = train.pm25
    fit, val = split_masks(week, args.fold)
    actual = y.loc[val]
    preds = pd.DataFrame({"actual": actual})
    if args.stage == "current":
        result = app.fit_peak_blend(peak.loc[fit], y.loc[fit], peak.loc[val])
        result.columns = ["LightGBM", "Peak features", "Peak weighting", CURRENT]
        preds = preds.join(result)
    elif args.stage == "features":
        baseline = lightgbm_fit(basic.loc[fit, BASE_COLS], y.loc[fit], BASELINE_PARAMS, 400)
        preds["Baseline"] = predict_lgb(baseline, basic.loc[val, BASE_COLS])
        features = lightgbm_fit(enhanced.loc[fit], y.loc[fit], STANDARD_PARAMS, 450)
        preds["Weather features"] = predict_lgb(features, enhanced.loc[val])
        score, config, rounds = select_lgb(enhanced.loc[fit], y.loc[fit], week.loc[fit])
        print(f"inner LightGBM RMSE {score:.4f}, config={config}, rounds={rounds}", flush=True)
        tuned = lightgbm_fit(enhanced.loc[fit], y.loc[fit], TRIAL_PARAMS[config], rounds)
        preds["Tuned LightGBM"] = predict_lgb(tuned, enhanced.loc[val])
    elif args.stage == "catboost":
        cat = cat_frame(basic)
        score, rounds = select_catboost(cat.loc[fit], y.loc[fit], week.loc[fit])
        print(f"inner CatBoost RMSE {score:.4f}, rounds={rounds}", flush=True)
        model = catboost_fit(cat.loc[fit], y.loc[fit], rounds)
        preds["CatBoost"] = model.predict(cat.loc[val]).clip(min=0)
    else:
        past, legacy = load_embassy(args.embassy_dir)
        for name, features in (("Embassy lags", past), ("Embassy future", legacy)):
            X = basic.join(features.reindex(train.datetime).reset_index(drop=True))
            model = lightgbm_fit(X.loc[fit], y.loc[fit], STANDARD_PARAMS, 450)
            preds[name] = predict_lgb(model, X.loc[val])
    print({name: round(app.rmse(actual, preds[name]), 6) for name in preds if name != "actual"}, flush=True)
    preds.to_csv(args.output)


def add_blends(preds):
    current, tuned, cat = (preds[name] for name in (CURRENT, "Tuned LightGBM", "CatBoost"))
    current_weight, tuned_weight, cat_weight = FORECAST_BLEND_WEIGHTS
    preds["Peak + tuned + CatBoost"] = current_weight * current + tuned_weight * tuned + cat_weight * cat
    preds["Peak + tuned"] = 0.5 * (current + tuned)
    if "Embassy lags" in preds:
        preds["Peak + Embassy lags"] = 0.5 * (current + preds["Embassy lags"])
    return preds


def submission(args, inputs):
    _, train, test, basic, basic_test, peak, peak_test, enhanced, enhanced_test, week = inputs
    y = train.pm25
    candidate = args.candidate
    if candidate in {"Embassy lags", "Peak + Embassy lags"}:
        past, _ = load_embassy(args.embassy_dir)
        X = basic.join(past.reindex(train.datetime).reset_index(drop=True))
        X_test = basic_test.join(past.reindex(test.datetime).reset_index(drop=True))
        result = predict_lgb(lightgbm_fit(X, y, STANDARD_PARAMS, 450), X_test)
    elif candidate == "CatBoost":
        cat, cat_test = cat_frame(basic), cat_frame(basic_test)
        _, rounds = select_catboost(cat, y, week)
        result = catboost_fit(cat, y, rounds).predict(cat_test).clip(min=0)
        print(f"CatBoost rounds={rounds}", flush=True)
    else:
        params, rounds = STANDARD_PARAMS, 450
        if candidate != "Weather features":
            _, config, rounds = select_lgb(enhanced, y, week)
            params = TRIAL_PARAMS[config]
        result = predict_lgb(lightgbm_fit(enhanced, y, params, rounds), enhanced_test)
        print(f"LightGBM leaves={params['num_leaves']}, rounds={rounds}", flush=True)
    if candidate.startswith("Peak +"):
        current = app.fit_peak_blend(peak, y, peak_test)["Оргил үеийн blend"].to_numpy()
        if candidate == "Peak + tuned + CatBoost":
            cat, cat_test = cat_frame(basic), cat_frame(basic_test)
            _, cat_rounds = select_catboost(cat, y, week)
            cat_prediction = catboost_fit(cat, y, cat_rounds).predict(cat_test).clip(min=0)
            current_weight, tuned_weight, cat_weight = FORECAST_BLEND_WEIGHTS
            result = current_weight * current + tuned_weight * result + cat_weight * cat_prediction
            print(f"CatBoost rounds={cat_rounds}", flush=True)
        else:
            result = 0.5 * (current + result)
    sample = pd.read_csv(ROOT / "Data/sample_submission.csv")
    assert list(sample.columns) == ["ID", "pm25"]
    assert test.ID.is_unique and sample.ID.is_unique and set(test.ID) == set(sample.ID)
    values = pd.Series(result, index=test.ID).reindex(sample.ID)
    assert np.isfinite(values).all() and values.ge(0).all()
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame({"ID": sample.ID, "pm25": values.to_numpy().round(2)}).to_csv(stream, index=False)
    print(f"{args.output}: {len(values)} predictions from {candidate}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-json", type=Path)
    parser.add_argument("--embassy-dir", type=Path)
    parser.add_argument("--fold", type=int, choices=range(6), default=0)
    parser.add_argument("--stage", choices=("current", "features", "catboost", "embassy", "submission"), required=True)
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "embassy" and args.embassy_dir is None:
        parser.error("--embassy-dir is required for Embassy experiments")
    if args.stage == "submission" and args.candidate is None:
        parser.error("--candidate is required for submission")
    if args.candidate in {"Embassy lags", "Peak + Embassy lags"} and args.embassy_dir is None:
        parser.error("--embassy-dir is required for an Embassy candidate")
    inputs = load_inputs(args.weather_json)
    if args.stage == "submission":
        submission(args, inputs)
    else:
        benchmark(args, inputs)


if __name__ == "__main__":
    main()
