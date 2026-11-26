"""Улаанбаатарын тухайн цагийн PM2.5-ыг LightGBM-ээр тооцно."""

import argparse
import json
import os
import ssl
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


DATA_DIR = Path(__file__).resolve().parent / "Data"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_TIMEZONE = "Asia/Ulaanbaatar"
WEATHER_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_speed_100m",
    "wind_direction_10m", "wind_gusts_10m", "surface_pressure", "precipitation",
    "boundary_layer_height", "shortwave_radiation", "cloud_cover",
]
STAGNATION_COLUMNS = [
    "era5_low_wind_hours_24h",
    "era5_low_wind_streak_hours",
    "era5_low_wind_and_shallow_layer_hours_24h",
]
MODEL_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
    "force_col_wise": True,
    "num_threads": 1,
}
PEAK_FEATURE_PREFIX = "era5_peak_"


def load_weather(start, end, weather_json=None):
    """Улаанбаатарын төвийн ERA5 мэдээллийг орон нутгийн цагаар уншина."""
    start_day = start.normalize() - pd.Timedelta(days=3)
    end_day = end.normalize()
    if weather_json is None:
        query = urlencode({
            "latitude": 47.918,
            "longitude": 106.917,
            "start_date": start_day.date(),
            "end_date": end_day.date(),
            "hourly": ",".join(WEATHER_COLUMNS),
            "timezone": WEATHER_TIMEZONE,
            "wind_speed_unit": "kmh",
            "models": "era5",
        })
        tls = ssl.create_default_context()
        system_ca = Path("/etc/ssl/cert.pem")
        if not tls.get_ca_certs() and system_ca.is_file():
            tls.load_verify_locations(cafile=str(system_ca))
        with urlopen(f"{WEATHER_URL}?{query}", timeout=90, context=tls) as response:
            payload = json.load(response)
    else:
        with weather_json.open(encoding="utf-8") as response:
            payload = json.load(response)

    if payload.get("timezone") != WEATHER_TIMEZONE:
        raise ValueError(f"Цаг агаарын цагийн бүс {WEATHER_TIMEZONE} байх ёстой.")
    units = payload["hourly_units"]
    if units["wind_speed_10m"] != "km/h" or units["boundary_layer_height"] != "m":
        raise ValueError("Салхины хурд км/ц, холилдлын давхаргын өндөр метрээр байх ёстой.")

    weather = pd.DataFrame(payload["hourly"])
    weather["time"] = pd.to_datetime(weather.time)
    weather = weather.set_index("time")[WEATHER_COLUMNS]
    if not weather.index.is_unique:
        raise ValueError("Open-Meteo өгөгдөлд давхардсан цаг байна.")
    hourly_index = pd.date_range(start_day, end_day + pd.Timedelta(hours=23), freq="h")
    if not hourly_index.isin(weather.index).all():
        raise ValueError("Open-Meteo өгөгдөл шаардлагатай бүх цагийг хамраагүй байна.")
    weather = weather.loc[hourly_index].copy()

    direction = np.deg2rad(weather.pop("wind_direction_10m"))
    weather["wind_direction_sin"] = np.sin(direction)
    weather["wind_direction_cos"] = np.cos(direction)
    weather["ventilation"] = weather.wind_speed_10m * weather.boundary_layer_height
    weather["pressure_change_24h"] = weather.surface_pressure.diff(24)
    for hours in (6, 24, 72):
        # Тухайн цаг болон түүнээс өмнөх цагуудыг нэгтгэнэ.
        history = weather[["wind_speed_10m", "boundary_layer_height", "ventilation"]].rolling(hours)
        for column, values in history.mean().items():
            weather[f"{column}_mean_{hours}h"] = values

    # 7.2 км/ц = 2 м/с. Хоосон хэмжилт дарааллыг тасална.
    wind = weather.wind_speed_10m
    layer = weather.boundary_layer_height
    low_wind = wind.le(7.2)
    shallow_layer = layer.le(200)
    weather["low_wind_hours_24h"] = low_wind.where(wind.notna()).rolling(24).sum()
    streak = low_wind.groupby((~low_wind).cumsum()).cumsum()
    weather["low_wind_streak_hours"] = streak.where(wind.notna())
    both = (low_wind & shallow_layer).where(wind.notna() & layer.notna())
    weather["low_wind_and_shallow_layer_hours_24h"] = both.rolling(24).sum()
    return weather.add_prefix("era5_")


def make_features(train, test, categories, weather=None):
    df = pd.concat([train.drop(columns="pm25"), test], ignore_index=True)
    df["time"] = pd.to_datetime(df.datetime)

    X = pd.DataFrame({
        "station": pd.Categorical(df.station, categories=categories),
        "hour": df.time.dt.hour,
        "dayofweek": df.time.dt.day_of_week,
        "month": df.time.dt.month,
        "year": df.time.dt.year,
        "dayofyear": df.time.dt.day_of_year,
        "is_heating": df.time.dt.month.isin([10, 11, 12, 1, 2, 3, 4]).astype(int),
        "temperature": df.temperature,
        "humidity": df.humidity,
    })

    g = df.groupby("time")
    X["city_temp"] = g.temperature.transform("mean")
    X["city_hum"] = g.humidity.transform("mean")
    X["temp_minus_city"] = df.temperature - X.city_temp

    df["_i"] = np.arange(len(df))
    s = df.sort_values(["station", "time"])
    for w in ["6h", "24h", "72h"]:
        r = s.groupby("station").rolling(w, on="time")
        X.loc[s._i.values, f"temp_mean_{w}"] = r.temperature.mean().values
        X.loc[s._i.values, f"temp_min_{w}"] = r.temperature.min().values
        X.loc[s._i.values, f"hum_mean_{w}"] = r.humidity.mean().values
    X["temp_change_24h"] = df.temperature - X["temp_mean_24h"]

    if weather is not None:
        # Станцуудын мөрийн дарааллыг хадгалж, цаг агаарын мэдээллийг цагаар тааруулна.
        hourly = weather.reindex(df.time)
        X = pd.concat([X, hourly.reset_index(drop=True)], axis=1)

    n = len(train)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def peak_weather_features(weather):
    features = pd.DataFrame(index=weather.index)
    cold = (-weather.era5_temperature_2m).clip(lower=0)
    for hours in (6, 24, 72):
        features[f"cold_mean_{hours}h"] = cold.rolling(hours).mean()
        features[f"precip_sum_{hours}h"] = weather.era5_precipitation.rolling(hours).sum()
        features[f"shallow_min_{hours}h"] = weather.era5_boundary_layer_height.rolling(hours).min()
    for hours in (6, 24):
        for column in ("temperature_2m", "wind_speed_10m", "boundary_layer_height", "surface_pressure"):
            features[f"{column}_delta_{hours}h"] = weather[f"era5_{column}"].diff(hours)
    return features.add_prefix(PEAK_FEATURE_PREFIX)


def fit_peak_blend(X, y, X_predict):
    """Оргил үеийн алдаанд илүү жин өгсөн загварыг хоёр энгийн загвартай холино."""
    params = {**MODEL_PARAMS, "learning_rate": 0.065, "max_bin": 95}
    weather_columns = X.columns[~X.columns.str.startswith(PEAK_FEATURE_PREFIX)]
    predictions = {}
    peak_weights = 1 + y.clip(upper=600) / 600
    for name, columns, weights in (
        ("ERA5 дан загвар", weather_columns, None),
        ("Нэмэлт цаг агаарын оролттой загвар", X.columns, None),
        ("Оргил хэмжилтийг жинлэсэн загвар", X.columns, peak_weights),
    ):
        training = lgb.Dataset(X[columns], label=y, weight=weights)
        model = lgb.train(params, training, num_boost_round=450)
        prediction = model.predict(X_predict[columns], num_threads=params["num_threads"])
        predictions[name] = np.maximum(prediction, 0)
    predictions["Оргил үеийн blend"] = (
        0.50 * predictions["ERA5 дан загвар"]
        + 0.25 * predictions["Нэмэлт цаг агаарын оролттой загвар"]
        + 0.25 * predictions["Оргил хэмжилтийг жинлэсэн загвар"]
    )
    return pd.DataFrame(predictions, index=X_predict.index)


def validate_peak_blend(X, y, week, validation_weeks):
    is_val = week.isin(validation_weeks)
    return fit_peak_blend(X.loc[~is_val], y.loc[~is_val], X.loc[is_val]), None


def predict_components(X, y, X_predict, params, rounds):
    raw_model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds[0])
    log_model = lgb.train(params, lgb.Dataset(X, label=np.log1p(y)), num_boost_round=rounds[1])
    threads = params["num_threads"]
    raw = np.maximum(raw_model.predict(X_predict, num_threads=threads), 0)
    logged = log_model.predict(X_predict, num_threads=threads)
    return raw, np.maximum(np.expm1(logged), 0)


def rmse(actual, prediction):
    return mean_squared_error(actual, prediction) ** 0.5


def log_model_pm25_rmse(prediction, dataset):
    actual = np.expm1(dataset.get_label())
    error = np.maximum(np.expm1(prediction), 0) - actual
    return "pm25_rmse", np.sqrt(np.mean(error ** 2)), False


def tune_model(X, y, week):
    """Тохиргоо, алхам, жинг гаднах validation-аас тусад нь сонгоно."""
    weeks = np.sort(week.unique())
    if len(weeks) < 2:
        raise ValueError("Тохиргоо сонгоход дор хаяж хоёр долоо хоног хэрэгтэй.")
    validation_weeks = np.random.default_rng(17).choice(
        weeks, size=max(1, len(weeks) // 5), replace=False
    )
    is_val = week.isin(validation_weeks)
    X_fit, X_val = X.loc[~is_val], X.loc[is_val]
    y_fit, y_val = y.loc[~is_val], y.loc[is_val]
    best_score = float("inf")

    for leaves, min_rows in [(63, 80), (95, 50)]:
        params = {**MODEL_PARAMS, "num_leaves": leaves, "min_data_in_leaf": min_rows}
        rounds = []
        predictions = []
        for use_log in (False, True):
            labels = np.log1p(y_fit) if use_log else y_fit
            val_labels = np.log1p(y_val) if use_log else y_val
            training = lgb.Dataset(X_fit, label=labels)
            model = lgb.train(
                {**params, "metric": "None" if use_log else "rmse"},
                training,
                num_boost_round=1400,
                valid_sets=[training.create_valid(X_val, label=val_labels)],
                feval=log_model_pm25_rmse if use_log else None,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            rounds.append(model.best_iteration)
            prediction = model.predict(X_val, num_threads=params["num_threads"])
            predictions.append(np.maximum(np.expm1(prediction) if use_log else prediction, 0))

        raw, logged = predictions
        difference = logged - raw
        length_squared = difference @ difference
        # Дотоод validation дээр квадрат алдааг хамгийн бага болгох log загварын жин.
        log_weight = np.clip((difference @ (y_val - raw)) / length_squared, 0, 1) if length_squared else 0.0
        score = rmse(y_val, raw + log_weight * difference)
        if score < best_score:
            best_score = score
            best_config = params, rounds, float(log_weight)
    return best_config


def fit_predict(X, y, X_predict, config=None):
    params, rounds, weight = config if config is not None else (MODEL_PARAMS, (800, 800), 0.2)
    raw, logged = predict_components(X, y, X_predict, params, rounds)
    return (1 - weight) * raw + weight * logged


def validate_models(X, y, week, validation_weeks):
    is_val = week.isin(validation_weeks)
    X_fit, X_val = X.loc[~is_val], X.loc[is_val]
    y_fit = y.loc[~is_val]
    raw, logged = predict_components(
        X_fit.drop(columns=STAGNATION_COLUMNS), y_fit, X_val.drop(columns=STAGNATION_COLUMNS),
        MODEL_PARAMS, (800, 800),
    )
    fixed_weather = fit_predict(X_fit, y_fit, X_val)
    config = tune_model(X_fit, y_fit, week.loc[~is_val])
    tuned = fit_predict(X_fit, y_fit, X_val, config)
    return pd.DataFrame({
        "ERA5 дан загвар": raw,
        "ERA5 тогтмол blend": 0.8 * raw + 0.2 * logged,
        "Сул салхины оролттой тогтмол blend": fixed_weather,
        "Тохируулсан blend": tuned,
    }, index=X_val.index), config


def report_errors(train, predictions, baseline, candidate):
    """OOF таамаглалын алдааг станц, сар, цаг, PM2.5-ийн түвшнээр задална."""
    errors = train[["station", "datetime", "pm25"]].copy()
    errors["сар"] = errors.datetime.dt.month
    errors["цаг"] = errors.datetime.dt.hour
    errors["PM2.5"] = pd.cut(
        errors.pm25, [-np.inf, 35, 75, 150, 300, np.inf],
        labels=["≤35", "35–75", "75–150", "150–300", ">300"],
    )
    errors["алдаа"] = predictions[candidate] - errors.pm25
    errors["квадрат"] = errors["алдаа"] ** 2
    errors["өмнөх квадрат"] = (predictions[baseline] - errors.pm25) ** 2
    total = errors["квадрат"].sum()
    for group in ("station", "сар", "цаг", "PM2.5"):
        table = errors.groupby(group, observed=True).agg(
            мөр=("pm25", "size"), bias=("алдаа", "mean"),
            өмнөх_MSE=("өмнөх квадрат", "mean"), MSE=("квадрат", "mean"),
            квадрат_алдаа=("квадрат", "sum"),
        )
        table["өмнөх RMSE"] = np.sqrt(table.pop("өмнөх_MSE"))
        table["RMSE"] = np.sqrt(table.pop("MSE"))
        table["алдааны хувь %"] = 100 * table.pop("квадрат_алдаа") / total if total else 0.0
        print(f"\nАлдааны задаргаа — {group}")
        print(table.round(2).to_string())


def evaluate_models(train, X, y, week, validate, baseline, candidate):
    # Өмнөх 47 долоо хоногийн хэмжилтийг хадгалж, мөн таван хуваалтаар шалгана.
    rng = np.random.default_rng(0)
    val_weeks = rng.choice(week.unique(), size=int(week.nunique() * 0.3), replace=False)
    folds = np.array_split(np.random.default_rng(42).permutation(np.sort(week.unique())), 5)
    labels = ["Өмнөх 47 долоо хоногийн хуваалт"] + [
        f"Долоо хоногийн хуваалт {number}/5" for number in range(1, 6)
    ]
    fold_predictions = []
    workers = min(4, os.cpu_count() or 1)
    print(f"Долоо хоногийн үнэлгээг {workers} зэрэгцээ ажлаар тооцоолж байна.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(partial(validate, X, y, week), [val_weeks, *folds])
        for label, (predictions, config) in zip(labels, results, strict=True):
            fold_predictions.append(predictions)
            print(label)
            if config is not None:
                params, rounds, weight = config
                print(
                    f"  leaves={params['num_leaves']}, min_data_in_leaf={params['min_data_in_leaf']}, "
                    f"rounds={rounds}, log_weight={weight:.3f}"
                )
            for name, prediction in predictions.items():
                print(f"  {name}: {rmse(y.loc[prediction.index], prediction):.2f}", flush=True)

    oof = pd.concat(fold_predictions[1:]).sort_index()
    print("5 хуваалтын нэгдсэн RMSE", flush=True)
    for name, prediction in oof.items():
        print(f"  {name}: {rmse(y, prediction):.2f}", flush=True)
    report_errors(train, oof, baseline, candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-json", type=Path, help="Өмнө нь татсан Open-Meteo архивын JSON")
    parser.add_argument("--model", choices=("peak", "tuned"), default="peak")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--validation-only", action="store_true", help="Зөвхөн validation ажиллуулна")
    run_mode.add_argument("--submission-only", action="store_true", help="Бүх train дээр сургаж submission үүсгэнэ")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["datetime"])
    train = train.sort_values(["station", "datetime"]).reset_index(drop=True)
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["datetime"])
    # LightGBM хоосон температурыг NaN байдлаар боловсруулна.
    start = min(train.datetime.min(), test.datetime.min())
    end = max(train.datetime.max(), test.datetime.max())
    weather = load_weather(start, end, args.weather_json)
    if args.model == "peak":
        weather = weather.join(peak_weather_features(weather))
        validate = validate_peak_blend
        baseline, candidate = "ERA5 дан загвар", "Оргил үеийн blend"
    else:
        validate = validate_models
        baseline, candidate = "ERA5 тогтмол blend", "Тохируулсан blend"
    X, X_test = make_features(train, test, sorted(train.station.unique()), weather)
    y = train.pm25

    week = train.datetime.dt.to_period("W")
    if not args.submission_only:
        evaluate_models(train, X, y, week, validate, baseline, candidate)
    if args.validation_only:
        return

    if args.model == "peak":
        prediction = fit_peak_blend(X, y, X_test)[candidate].to_numpy()
    else:
        config = tune_model(X, y, week)
        prediction = fit_predict(X, y, X_test, config)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", usecols=["ID"])
    submission = sample.merge(
        pd.DataFrame({"ID": test.ID, "pm25": prediction.round(2)}),
        on="ID",
        how="left",
        validate="one_to_one",
    )
    if len(submission) != len(test) or submission.pm25.isna().any():
        raise ValueError("Test болон sample_submission файлуудын ID таарахгүй байна.")
    submission.to_csv(args.output, index=False)
    print(f"{args.output}: {len(submission)} таамаглал")


if __name__ == "__main__":
    main()
