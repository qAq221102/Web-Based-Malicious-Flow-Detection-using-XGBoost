import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
import os

import plot_util

# === 設定路徑 ===
validset_path = "final_data/final_val_cleaned_binary.csv"
target_dir = "target"
saved_model_path = f"{target_dir}/xgb_model.pkl"
report_path = f"{target_dir}/classification_report.txt"

# 自動建立目錄（若不存在）
os.makedirs(target_dir, exist_ok=True)

# === 檢查資料（可選）===


def check_data(df, name="Data"):
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.to_numpy()).sum()
    print(f"\n--- check {name} ---")
    print(f"numbers of NaN: {nan_count}")
    print(f"numbers of inf: {inf_count}")


# === 載入資料與模型 ===
df = pd.read_csv(validset_path)
# check_data(df, "Validate")
X_val = df.drop(columns=["Label"])
y_val = df["Label"]

model = joblib.load(saved_model_path)

# === 預測與評估 ===
y_pred = model.predict(X_val)
report = classification_report(y_val, y_pred, zero_division=0,digits=4)

# 儲存報告
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"classification report is saved to: {report_path}")

print("\n=== Classification Report ===")
print(report)

# # === 混淆矩陣（二分類）===
plot_util.plot_confusion_matrix(
    y_val, y_pred)
