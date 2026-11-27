# Baseline: Улаанбаатарын PM2.5-ыг таамаглах
#
# Энэ notebook бол тэмцээний анхны суурь шийдэл (baseline). Энэ үр дүнгээс илүү оноо авах нь та бүхний зорилго.
#
# Даалгавар: станц, огноо/цаг, температур, чийгшлийн мэдээллээр тухайн цагийн PM2.5 агууламжийг (µg/m³) таамаглах.
# Үнэлгээ: RMSE, бага байх тусам сайн.

import os, glob, json, urllib.request
import catboost as cb
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

# ## Тохиргоо
RUN_CHECKS = False     # Week-based CV + feature тест (удаан). Шийдвэрээ гаргасны дараа False болгоно
RUN_TUNING = False    # Hyperparameter хайлт (10-30 минут). True болгож ажиллуулна
RUN_CATBOOST = False  # CatBoost-той харьцуулах. True болгож ажиллуулна
DROP = []             # CV тестийн дараа хасах feature-үүдээ энд бичнэ, жишээ нь ["is_weekend", "very_cold"]

# АНУ-ын ЭСЯ-ны PM2.5 (гадны дата). Data/embassy дотор 5 файл байна
def load_embassy():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "embassy", "*.csv")))
    e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    e = e[(e["QC Name"] == "Valid") & (e["Raw Conc."] >= 0)]   # Invalid, Missing утгыг хасна
    e["time"] = pd.to_datetime(e[["Year", "Month", "Day", "Hour"]].rename(columns=str.lower))
    s = e.groupby("time")["Raw Conc."].mean()
    s = s.reindex(pd.date_range("2018-12-28", "2023-02-01", freq="h"))

    out = pd.DataFrame(index=s.index)
    # ЭСЯ-ны хэмжилт манай дататай ~2 цагийн зөрүүтэй тул t-1 ... t+4 цагийн утгыг авна
    for k in [-1, 0, 1, 2, 3, 4]:
        out[f"emb_t{k:+d}"] = s.shift(-k)
    out["emb_mean_t0_t4"] = s.shift(-4).rolling(5, min_periods=1).mean()
    out["emb_mean_12h"] = s.rolling(12, center=True, min_periods=1).mean()
    out["emb_mean_24h"] = s.rolling(24, center=True, min_periods=1).mean()
    out["emb_max_24h"] = s.rolling(24, center=True, min_periods=1).max()
    out["emb_mean_t0_t4_6h"] = s.shift(-2).rolling(6, center=True, min_periods=1).mean()
    out["emb_diff_1h"] = s.shift(-2) - s.shift(-1)          # өсөж байна уу, буурч байна уу
    out["emb_mean_72h"] = s.rolling(72, center=True, min_periods=1).mean()
    out["emb_missing"] = s.shift(-2).isna().astype(int)
    return out

# Open-Meteo ERA5 цаг агаар (гадны дата). Эхний удаа татаж Data/weather.json-д хадгална
WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_speed_100m",
                "wind_direction_10m", "wind_gusts_10m", "surface_pressure", "precipitation",
                "boundary_layer_height", "shortwave_radiation", "cloud_cover"]

def load_weather():
    path = os.path.join(DATA_DIR, "weather.json")
    if not os.path.exists(path):
        url = ("https://archive-api.open-meteo.com/v1/archive?latitude=47.918&longitude=106.917"
            "&start_date=2018-12-28&end_date=2023-02-01&hourly=" + ",".join(WEATHER_COLS) +
            "&timezone=Asia%2FUlaanbaatar&wind_speed_unit=kmh&models=era5")
        urllib.request.urlretrieve(url, path)
    w = pd.DataFrame(json.load(open(path, encoding="utf-8"))["hourly"])
    w["time"] = pd.to_datetime(w.time)
    w = w.set_index("time")

    # Салхины чиглэлийг вектор болгоно (359° ба 1° ойрхон гэдгийг загвар ойлгоно)
    rad = np.deg2rad(w.pop("wind_direction_10m"))
    w["wind_x"] = w.wind_speed_10m * np.cos(rad)
    w["wind_y"] = w.wind_speed_10m * np.sin(rad)
    # Агаар сэлгэлт = салхи × холилдлын давхаргын өндөр. Бага бол утаа хуримтлагдана
    w["ventilation"] = w.wind_speed_10m * w.boundary_layer_height
    w["wind_shear"] = w.wind_speed_100m - w.wind_speed_10m
    w["pressure_change_24h"] = w.surface_pressure.diff(24)

    # Түүх: сүүлийн N цагийн дундаж / минимум
    for h in [6, 24, 72]:
        for c in ["wind_speed_10m", "boundary_layer_height", "ventilation"]:
            w[f"{c}_mean_{h}h"] = w[c].rolling(h).mean()
    for h in [3, 12, 48]:
        w[f"ventilation_min_{h}h"] = w.ventilation.rolling(h).min()
        w[f"temp_mean_{h}h"] = w.temperature_2m.rolling(h).mean()
    w["wind_x_mean_12h"] = w.wind_x.rolling(12).mean()
    w["wind_y_mean_12h"] = w.wind_y.rolling(12).mean()

    # Салхигүй үе: 7.2 км/ц = 2 м/с-ээс сул салхи хэдэн цаг үргэлжилсэн бэ
    low_wind = (w.wind_speed_10m <= 7.2).astype(int)
    w["low_wind_hours_24h"] = low_wind.rolling(24).sum()
    w["low_wind_streak"] = low_wind.groupby((low_wind == 0).cumsum()).cumsum()
    w["low_wind_shallow_24h"] = (low_wind * (w.boundary_layer_height <= 200)).rolling(24).sum()
    return w.add_prefix("era5_")

# Kaggle дээр тэмцээний дата /kaggle/input доор байрлана
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")

# Validation долоо хоногуудыг clean()-ээс өмнө сонгоно (мөрийн эрэмбэ өөрчлөгдөхөөс өмнө)
week0 = pd.to_datetime(train.datetime).dt.to_period("W").astype(str)
rng = np.random.default_rng(0)
val_weeks = set(rng.choice(week0.unique(), size=int(week0.nunique() * 0.3), replace=False))
print(train.shape, test.shape)
print(train.head())

# ## 1. Датаг товч харах

print(train.describe().round(1))
print("\nСтанц бүрийн дундаж PM2.5:")
print(train.groupby("station").pm25.mean().round(1).sort_values())

# Цаг бүрийн дундаж PM2.5: өвөл, зун хоёрыг харьцуулъя
d = pd.to_datetime(train.datetime)
season = np.where(d.dt.month.isin([11, 12, 1, 2]), "Өвөл (11-2 сар)", "Бусад сар")
print(train.groupby([season, d.dt.hour]).pm25.mean().unstack(0).round(0))

# Dataniihaa medeelliig avah tegeed aldaag olj zasah (data cleaning)
# Number of rows
print(train.shape, test.shape)

# 1. Duplicate station + hour
print("Давхардсан (train):", train.duplicated(["station", "datetime"]).sum())
print("Давхардсан (test):", test.duplicated(["station", "datetime"]).sum())

# 2. Broken datetimes (ones pandas can't read)
print("Буруу огноо (train):", pd.to_datetime(train.datetime, errors="coerce").isna().sum())
print("Буруу огноо (test):", pd.to_datetime(test.datetime, errors="coerce").isna().sum())

# 3. Station names (look for typos / extra spaces)
print(train.station.value_counts())
print(test.station.value_counts())

# 4. Empty values in every column
print(train.isna().sum())
print(test.isna().sum())

# 5. Ranges: min / max of temperature, humidity, pm25
print(train[["temperature", "humidity", "pm25"]].describe().round(1))
print(test[["temperature", "humidity"]].describe().round(1))

# 6. PM2.5 extremes
print("PM2.5 > 1000:", (train.pm25 > 1000).sum())
print("PM2.5 = 0:", (train.pm25 == 0).sum())

# 7. Which stations have no latitude/longitude
print(train[train.latitude.isna()].station.value_counts())
print(test[test.latitude.isna()].station.value_counts())

#ym nemsens

def clean(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df.datetime)
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    n_before = df.temperature.isna().sum()
    # Hooson datag oiirhon tsagtai baigaa datagaar nuhuh
    # Zuvhun temperature dotor ni missing data baisan baina eniig zassan.
    df["temperature"] = (df.set_index("datetime").groupby("station")["temperature"]
                        .transform(lambda s: s.interpolate(method="time", limit=3, limit_area="inside")).values)
    print(f"Хоосон температур: {n_before} → {df.temperature.isna().sum()}")
    return df

train = clean(train)
test = clean(test)


# ## 2. Feature бэлдэх
#
# Baseline-д зөвхөн хамгийн энгийн feature-үүдийг ашиглана: станц, цаг, гараг, сар, температур, чийгшил.


def make_features(train, test, categories, embassy=None, weather=None):
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
        "temp_missing": df.temperature.isna().astype(int),
        "is_weekend": (df.time.dt.day_of_week >= 5).astype(int),
        "is_peak_hour": df.time.dt.hour.isin([8, 9, 10, 19, 20, 21, 22, 23]).astype(int),
        "very_cold": (df.temperature < -20).astype(int),
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

    if embassy is not None:
        X = pd.concat([X, embassy.reindex(df.time).reset_index(drop=True)], axis=1)
    
    if weather is not None:
        X = pd.concat([X, weather.reindex(df.time).reset_index(drop=True)], axis=1)
        # Станцын хэмжилт ба ERA5-ийн зөрүү: станц хотын дунджаас хүйтэн/чийглэг үү
        X["temp_minus_era5"] = X.temperature - X.era5_temperature_2m
        X["hum_minus_era5"] = X.humidity - X.era5_relative_humidity_2m
    
    n = len(train)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)

stations = sorted(train.station.unique())
embassy = load_embassy()
weather = load_weather()
X, X_test = make_features(train, test, stations, embassy, weather)
y = train.pm25

# CV тестээр хэрэггүй гэж гарсан feature-үүдийг хасна (DROP-ийг дээр тохируулна)
X, X_test = X.drop(columns=DROP), X_test.drop(columns=DROP)

# ## 3. Validation
#
# Test нь санамсаргүй сонгосон бүтэн долоо хоногуудаас бүрддэг. Тиймээс validation-ийг мөн долоо хоногоор нь салгаж хийнэ.
# Мөр бүрээр санамсаргүй хуваавал хөрш цагуудын утга хоорондоо хэт төстэй байдаг тул оноо худлаа сайн гарна.

week = pd.to_datetime(train.datetime).dt.to_period("W").astype(str)
is_val = week.isin(val_weeks).values

params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)

# Early stopping: validation алдаа 100 мод дараалан сайжрахгүй бол зогсооно
def early_stop(params):
    dtr = lgb.Dataset(X[~is_val], y[~is_val])
    dva = lgb.Dataset(X[is_val], y[is_val], reference=dtr)
    m = lgb.train({**params, "metric": "rmse"}, dtr, num_boost_round=5000,
                valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    p = np.clip(m.predict(X[is_val], num_iteration=m.best_iteration), 0, None)
    p_tr = np.clip(m.predict(X[~is_val], num_iteration=m.best_iteration), 0, None)
    val_rmse = mean_squared_error(y[is_val], p) ** 0.5
    print("Хамгийн сайн модны тоо:", m.best_iteration)
    print("Train RMSE:", round(mean_squared_error(y[~is_val], p_tr) ** 0.5, 2))
    print("Validation RMSE:", round(val_rmse, 2))
    return m, m.best_iteration, val_rmse

m, best_rounds, val_rmse = early_stop(params)
print("Харьцуулалт, бүгдэд нь train-ийн дунджийг таамагласан үед:",
    round(mean_squared_error(y[is_val], np.full(is_val.sum(), y[~is_val].mean())) ** 0.5, 2))

# Week-based cross validation: долоо хоног бүр нэг бүлэг
groups = week.values
gkf = GroupKFold(n_splits=5)

def cv_rmse(X, rounds):
    scores = []
    for tr_i, va_i in gkf.split(X, y, groups):
        mm = lgb.train(params, lgb.Dataset(X.iloc[tr_i], y.iloc[tr_i]), num_boost_round=rounds)
        pp = np.clip(mm.predict(X.iloc[va_i]), 0, None)
        scores.append(mean_squared_error(y.iloc[va_i], pp) ** 0.5)
    print(f"CV RMSE = {np.mean(scores):.2f} (±{np.std(scores):.2f})")
    return scores

def compare(old, new):
    # Ижил fold-ууд дээр хоёр хувилбарын ялгаа (paired), 2×SE дүрэм
    d = np.array(new) - np.array(old)
    se = d.std(ddof=1) / np.sqrt(len(d))
    verdict = "бодит өөрчлөлт" if abs(d.mean()) > 2 * se else "азаас болсон байж магадгүй"
    print(f"Өөрчлөлт {d.mean():+.2f} RMSE (SE {se:.2f}) → {verdict}")

if RUN_CHECKS:
    base_scores = cv_rmse(X, best_rounds)

    # Feature importance ба SHAP
    gain = pd.Series(m.feature_importance(importance_type="gain"), index=X.columns)
    shap = m.predict(X[is_val], num_iteration=best_rounds, pred_contrib=True)[:, :-1]   # сүүлийн багана = bias
    shap_imp = pd.Series(np.abs(shap).mean(axis=0), index=X.columns)
    print(pd.DataFrame({"gain": gain, "дундаж |SHAP|": shap_imp}).sort_values("дундаж |SHAP|").round(1))

    # Тест 1: хамгийн сул 3 feature-ийг хасвал?
    weak = shap_imp.sort_values().index[:3].tolist()
    print("Тест 1, хасах:", weak)
    compare(base_scores, cv_rmse(X.drop(columns=weak), best_rounds))

    # Тест 2: шинэ 4 feature хэрэгтэй юу?
    new_flags = [c for c in ["temp_missing", "is_weekend", "is_peak_hour", "very_cold"] if c in X.columns]
    if new_flags:
        print("Тест 2, хасах:", new_flags)
        compare(base_scores, cv_rmse(X.drop(columns=new_flags), best_rounds))

    # Унших дүрэм: "+... → бодит өөрчлөлт" гарвал хасах нь муу, үлдээнэ.
    # Бусад тохиолдолд (сөрөг эсвэл "азаас болсон") тэр feature-үүдийг дээрх DROP-д нэмнэ.

# Hyperparameter хайлт (week-based CV-ээр)
if RUN_TUNING:
    search = RandomizedSearchCV(
        lgb.LGBMRegressor(n_estimators=best_rounds, learning_rate=params["learning_rate"],
                        subsample_freq=1, n_jobs=1, random_state=42, verbose=-1),
        {"num_leaves": randint(15, 128), "min_child_samples": randint(20, 200),
        "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-3, 10)},
        n_iter=12, cv=gkf, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
    search.fit(X, y, groups=groups)
    print("Tuning CV RMSE:", round(-search.best_score_, 2))
    bp = search.best_params_
    new_params = {**params, "num_leaves": bp["num_leaves"], "min_data_in_leaf": bp["min_child_samples"],
                "bagging_fraction": bp["subsample"], "feature_fraction": bp["colsample_bytree"],
                "lambda_l2": bp["reg_lambda"]}
    m2, br2, v2 = early_stop(new_params)   # шинэ тохиргоогоор модны тоог дахин олно
    if v2 < val_rmse:
        params, m, best_rounds, val_rmse = new_params, m2, br2, v2
        print("Шинэ params-ийг авлаа:", params)
    else:
        print("Tuning сайжруулаагүй, хуучин params-аа үлдээв")

# CatBoost-той харьцуулах
if RUN_CATBOOST:
    import catboost as cb
    Xb = X.copy(); Xb["station"] = Xb["station"].astype(str)
    cbm = cb.CatBoostRegressor(iterations=3000, learning_rate=0.05, depth=6, loss_function="RMSE",
                            cat_features=["station"], early_stopping_rounds=100, random_seed=42, verbose=0)
    cbm.fit(Xb[~is_val], y[~is_val], eval_set=(Xb[is_val], y[is_val]))
    p_cb = np.clip(cbm.predict(Xb[is_val]), 0, None)
    print("CatBoost validation RMSE:", round(mean_squared_error(y[is_val], p_cb) ** 0.5, 2))
    print("LightGBM validation RMSE:", round(val_rmse, 2))

# ## 4. Бүх train дээр сургаж submission үүсгэх

# model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=best_rounds)
# submission = pd.DataFrame({"ID": test.ID, "pm25": np.clip(model.predict(X_test), 0, None).round(2)})
# submission.to_csv("submission.csv", index=False)
# print(submission.head())
    
def lgb_fit_predict(X_tr, y_tr, X_pr, rounds, log=False, peak=False, seed=42):
    yt = np.log1p(y_tr) if log else y_tr
    # Оргил утгад илүү жин: RMSE том алдаанд хамгийн мэдрэмтгий
    w = (1 + y_tr.clip(upper=600) / 600) if peak else None
    mm = lgb.train({**params, "seed": seed}, lgb.Dataset(X_tr, yt, weight=w), num_boost_round=rounds)
    p = mm.predict(X_pr)
    return np.clip(np.expm1(p) if log else p, 0, None)

def cb_fit_predict(X_tr, y_tr, X_pr, rounds):
    Xa, Xp = X_tr.copy(), X_pr.copy()
    Xa["station"] = Xa["station"].astype(str); Xp["station"] = Xp["station"].astype(str)
    mm = cb.CatBoostRegressor(iterations=rounds, learning_rate=0.05, depth=6, cat_features=["station"],
                              random_seed=42, verbose=0)
    return np.clip(mm.fit(Xa, y_tr).predict(Xp), 0, None)

def blend_predict(X_tr, y_tr, X_pr, cb_rounds, rounds):
    preds = {
        "lgb":      np.mean([lgb_fit_predict(X_tr, y_tr, X_pr, rounds, seed=s) for s in [1, 2, 3]], axis=0),
        "lgb_peak": lgb_fit_predict(X_tr, y_tr, X_pr, rounds, peak=True),
        "lgb_log":  lgb_fit_predict(X_tr, y_tr, X_pr, rounds, log=True),
        # "cat":      cb_fit_predict(X_tr, y_tr, X_pr, cb_rounds),
    }
    return pd.DataFrame(preds)

# ## Blend-ийг 5-fold CV-ээр шалгах (Kaggle submission хэмнэх зорилготой)
RUN_BLEND_CV = True

if RUN_BLEND_CV:
    oof = pd.DataFrame(0.0, index=X.index, columns=["lgb", "lgb_peak", "lgb_log"])
    for tr_i, va_i in gkf.split(X, y, groups):
        Xtr, ytr, Xva = X.iloc[tr_i], y.iloc[tr_i], X.iloc[va_i]
        oof.iloc[va_i, 0] = lgb_fit_predict(Xtr, ytr, Xva, best_rounds)
        oof.iloc[va_i, 1] = lgb_fit_predict(Xtr, ytr, Xva, best_rounds, peak=True)
        oof.iloc[va_i, 2] = lgb_fit_predict(Xtr, ytr, Xva, best_rounds, log=True)

    rmse = lambda p: mean_squared_error(y, p) ** 0.5
    for c in oof:
        print(f"CV {c:10s} RMSE: {rmse(oof[c]):.2f}")

    # Хэд хэдэн жинг туршина. Хамгийн бага RMSE-тэйг сонгоно
    candidates = {
        "зөвхөн lgb":          {"lgb": 1.0, "lgb_peak": 0.0,  "lgb_log": 0.0},
        "lgb + peak":          {"lgb": 0.6, "lgb_peak": 0.4,  "lgb_log": 0.0},
        "lgb + peak + log":    {"lgb": 0.6, "lgb_peak": 0.25, "lgb_log": 0.15},
        "lgb + log":           {"lgb": 0.8, "lgb_peak": 0.0,  "lgb_log": 0.2},
    }
    for name, wts in candidates.items():
        print(f"CV {name:18s} RMSE: {rmse(sum(oof[c] * w for c, w in wts.items())):.2f}")

# Validation дээр загвар бүр болон холимог хэр байгааг харна
cb_rounds = cbm.get_best_iteration() if RUN_CATBOOST else 1500
P = blend_predict(X[~is_val], y[~is_val], X[is_val], cb_rounds, best_rounds)
for c in P:
    print(f"{c:10s} RMSE:", round(mean_squared_error(y[is_val], P[c]) ** 0.5, 2))
WEIGHTS = {"lgb": 0.4, "lgb_peak": 0.25, "lgb_log": 0.1,}
p_blend = sum(P[c] * w for c, w in WEIGHTS.items())
print("Blend RMSE:", round(mean_squared_error(y[is_val], p_blend) ** 0.5, 2))

# Бүх train дээр сургаж submission
P_test = blend_predict(X, y, X_test, cb_rounds, best_rounds)
pm = sum(P_test[c] * w for c, w in WEIGHTS.items())
submission = pd.DataFrame({"ID": test.ID, "pm25": pm.round(2).values})
submission.to_csv("submission.csv", index=False)
print(submission.head())

    # ## Сайжруулах санаанууд
    # - Температурын түүх: сүүлийн 6/24/72 цагийн дундаж, хамгийн бага температур, температурын өөрчлөлт. Хүйтэн, салхигүй шөнө утаа хуримтлагддаг.
    # - Хотын түвшний feature: тухайн цагт бүх станцын дундаж температур, чийгшил.
    # - Он (`year`): сенсорын түвшин, хотын нөхцөл жил бүр өөрчлөгддөг.
    # - Target хувиргалт: `log1p(pm25)` дээр сургаад буцааж хувиргах, хэд хэдэн загварыг холих (blending).
    # - Гадны дата: цаг агаарын нэмэлт мэдээлэл, жишээ нь Open-Meteo-ийн салхи, даралт.