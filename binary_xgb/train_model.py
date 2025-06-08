import pandas as pd
import numpy as np
import joblib
from model import build_binary_xgb

trainset_path = "final_data/final_train_cleaned_binary.csv"
saved_model_path = "target/xgb_model.pkl"


def check_data(df, name="Data"):
    """
    檢查資料集是否含有 NaN 或 inf 值
    """
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.to_numpy()).sum()

    print(f"\n--- check {name} ---")
    print(f"numbers of NaN: {nan_count}")
    print(f"numbers of inf: {inf_count}")


# 載入資料
df = pd.read_csv(trainset_path)
# check_data(df, 'Train')
X = df.drop(columns=["Label"])
y = df["Label"]


# 建立與訓練模型
model = build_binary_xgb()
model.fit(X, y)

# 儲存模型
joblib.dump(model, saved_model_path)
print(f"model is saved to: {saved_model_path}")
