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

# Kaggle дээр тэмцээний дата /kaggle/input доор байрлана
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
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
        "dayofweek": df.time.dt.dayofweek,
        "month": df.time.dt.month,
        "year": df.time.dt.year,
        "dayofyear": df.time.dt.dayofyear,
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

    n = len(train)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)

stations = sorted(train.station.unique())
X, X_test = make_features(train, test, stations)
y = train.pm25

# ## 3. Validation
#
# Test нь санамсаргүй сонгосон бүтэн долоо хоногуудаас бүрддэг. Тиймээс validation-ийг мөн долоо хоногоор нь салгаж хийнэ.
# Мөр бүрээр санамсаргүй хуваавал хөрш цагуудын утга хоорондоо хэт төстэй байдаг тул оноо худлаа сайн гарна.

week = pd.to_datetime(train.datetime).dt.to_period("W").astype(str)
rng = np.random.default_rng(0)
val_weeks = set(rng.choice(week.unique(), size=int(week.nunique() * 0.3), replace=False))
is_val = week.isin(val_weeks).values

params = dict(objective="regression", learning_rate=0.005, num_leaves=10, min_data_in_leaf=120,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42)
m = lgb.train(params, lgb.Dataset(X[~is_val], y[~is_val]), num_boost_round=400)
p = np.clip(m.predict(X[is_val]), 0, None)
print("Validation RMSE:", round(mean_squared_error(y[is_val], p) ** 0.5, 2))
print("Харьцуулалт, бүгдэд нь train-ийн дунджийг таамагласан үед:",
      round(mean_squared_error(y[is_val], np.full(is_val.sum(), y[~is_val].mean())) ** 0.5, 2))

# ## 4. Бүх train дээр сургаж submission үүсгэх

model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=400)
submission = pd.DataFrame({"ID": test.ID, "pm25": np.clip(model.predict(X_test), 0, None).round(2)})
submission.to_csv("submission.csv", index=False)
print(submission.head())

# ## Сайжруулах санаанууд
# - Температурын түүх: сүүлийн 6/24/72 цагийн дундаж, хамгийн бага температур, температурын өөрчлөлт. Хүйтэн, салхигүй шөнө утаа хуримтлагддаг.
# - Хотын түвшний feature: тухайн цагт бүх станцын дундаж температур, чийгшил.
# - Он (`year`): сенсорын түвшин, хотын нөхцөл жил бүр өөрчлөгддөг.
# - Target хувиргалт: `log1p(pm25)` дээр сургаад буцааж хувиргах, хэд хэдэн загварыг холих (blending).
# - Гадны дата: цаг агаарын нэмэлт мэдээлэл, жишээ нь Open-Meteo-ийн салхи, даралт.
