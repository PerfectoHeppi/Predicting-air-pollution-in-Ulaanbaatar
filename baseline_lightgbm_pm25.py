# Baseline: Улаанбаатарын PM2.5-ыг таамаглах
#
# Энэ notebook бол тэмцээний анхны суурь шийдэл (baseline). Энэ үр дүнгээс илүү оноо авах нь та бүхний зорилго.
#
# Даалгавар: станц, огноо/цаг, температур, чийгшлийн мэдээллээр тухайн цагийн PM2.5 агууламжийг (µg/m³) таамаглах.
# Үнэлгээ: RMSE, бага байх тусам сайн.

import os, glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

# ## Тохиргоо
RUN_CHECKS = True     # Week-based CV + feature тест (удаан). Шийдвэрээ гаргасны дараа False болгоно
RUN_TUNING = False    # Hyperparameter хайлт (10-30 минут). True болгож ажиллуулна
RUN_CATBOOST = True  # CatBoost-той харьцуулах. True болгож ажиллуулна
DROP = []             # CV тестийн дараа хасах feature-үүдээ энд бичнэ, жишээ нь ["is_weekend", "very_cold"]

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


def make_features(train, test, categories):
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

    n = len(train)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)

stations = sorted(train.station.unique())
X, X_test = make_features(train, test, stations)
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

model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=best_rounds)
submission = pd.DataFrame({"ID": test.ID, "pm25": np.clip(model.predict(X_test), 0, None).round(2)})
submission.to_csv("submission.csv", index=False)
print(submission.head())

# ## Сайжруулах санаанууд
# - Температурын түүх: сүүлийн 6/24/72 цагийн дундаж, хамгийн бага температур, температурын өөрчлөлт. Хүйтэн, салхигүй шөнө утаа хуримтлагддаг.
# - Хотын түвшний feature: тухайн цагт бүх станцын дундаж температур, чийгшил.
# - Он (`year`): сенсорын түвшин, хотын нөхцөл жил бүр өөрчлөгддөг.
# - Target хувиргалт: `log1p(pm25)` дээр сургаад буцааж хувиргах, хэд хэдэн загварыг холих (blending).
# - Гадны дата: цаг агаарын нэмэлт мэдээлэл, жишээ нь Open-Meteo-ийн салхи, даралт.