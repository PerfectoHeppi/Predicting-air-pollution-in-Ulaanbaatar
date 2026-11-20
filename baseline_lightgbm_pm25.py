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
DATA_DIR = os.environ.get("DATA_DIR") or os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
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

# ## 2. Feature бэлдэх
#
# Baseline-д зөвхөн хамгийн энгийн feature-үүдийг ашиглана: станц, цаг, гараг, сар, температур, чийгшил.


def make_features(df, categories=None):
    d = pd.to_datetime(df.datetime)
    X = pd.DataFrame({
        "station": pd.Categorical(df.station, categories=categories),
        "hour": d.dt.hour,
        "dayofweek": d.dt.dayofweek,
        "month": d.dt.month,
        "temperature": df.temperature,
        "humidity": df.humidity,
    })
    return X


stations = sorted(train.station.unique())
X = make_features(train, stations)
y = train.pm25
X_test = make_features(test, stations)

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
