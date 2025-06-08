import os
import dask.dataframe as dd
import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder
import numpy as np
idx_datatypes = {
    'Dst Port': np.int64,
    'Protocol': np.int8,
    'Timestamp': object,
    'Flow Duration': np.int64,
    'Tot Fwd Pkts': np.int16,
    'Tot Bwd Pkts': np.int16,
    'TotLen Fwd Pkts': np.int32,
    'TotLen Bwd Pkts': np.int32,
    'Fwd Pkt Len Max': np.int32,
    'Fwd Pkt Len Min': np.int32,
    'Fwd Pkt Len Mean': np.float64,
    'Fwd Pkt Len Std': np.float64,
    'Bwd Pkt Len Max': np.float64,
    'Bwd Pkt Len Min': np.int16,
    'Bwd Pkt Len Mean': np.float64,
    'Bwd Pkt Len Std': np.float64,
    'Flow Byts/s': np.float64,
    'Flow Pkts/s': np.float64,
    'Flow IAT Mean': np.float64,
    'Flow IAT Std': np.float64,
    'Flow IAT Max': np.int64,
    'Flow IAT Min': np.int32,
    'Fwd IAT Tot': np.int32,
    'Fwd IAT Mean': np.float32,
    'Fwd IAT Std': np.float64,
    'Fwd IAT Max': np.int32,
    'Fwd IAT Min': np.int32,
    'Bwd IAT Tot': np.int32,
    'Bwd IAT Mean': np.float64,
    'Bwd IAT Std': np.float64,
    'Bwd IAT Max': np.int64,
    'Bwd IAT Min': np.int64,
    'Fwd PSH Flags': np.int8,
    'Bwd PSH Flags': np.int8,
    'Fwd URG Flags': np.int8,
    'Bwd URG Flags': np.int8,
    'Fwd Header Len': np.int32,
    'Bwd Header Len': np.int32,
    'Fwd Pkts/s': np.float64,
    'Bwd Pkts/s': np.float64,
    'Pkt Len Min': np.int16,
    'Pkt Len Max': np.int32,
    'Pkt Len Mean': np.float64,
    'Pkt Len Std': np.float64,
    'Pkt Len Var': np.float64,
    'FIN Flag Cnt': np.int8,
    'SYN Flag Cnt': np.int8,
    'RST Flag Cnt': np.int8,
    'PSH Flag Cnt': np.int8,
    'ACK Flag Cnt': np.int8,
    'URG Flag Cnt': np.int8,
    'CWE Flag Count': np.int8,
    'ECE Flag Cnt': np.int8,
    'Down/Up Ratio': np.int16,
    'Pkt Size Avg': np.float32,
    'Fwd Seg Size Avg': np.float32,
    'Bwd Seg Size Avg': np.float32,
    'Fwd Byts/b Avg': np.int8,
    'Fwd Pkts/b Avg': np.int8,
    'Fwd Blk Rate Avg': np.int8,
    'Bwd Byts/b Avg': np.int8,
    'Bwd Pkts/b Avg': np.int8,
    'Bwd Blk Rate Avg': np.int8,
    'Subflow Fwd Pkts': np.int16,
    'Subflow Fwd Byts': np.int32,
    'Subflow Bwd Pkts': np.int16,
    'Subflow Bwd Byts': np.int32,
    'Init Fwd Win Byts': np.int32,
    'Init Bwd Win Byts': np.int32,
    'Fwd Act Data Pkts': np.int16,
    'Fwd Seg Size Min': np.int8,
    'Active Mean': np.float64,
    'Active Std': np.float64,
    'Active Max': np.int32,
    'Active Min': np.int32,
    'Idle Mean': np.float64,
    'Idle Std': np.float64,
    'Idle Max': np.int64,
    'Idle Min': np.int64,
    'Label': object
}


def merge_data_by_labels():
    # === 設定參數 ===
    input_folders = ["CICIDS2017_improved", "CSECICIDS2018_improved"]
    output_folder = "labeled_data"
    time_column = "Timestamp"
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    label_column = "Label"
    drop_columns = [
        'id', 'Flow ID', 'Attempted Category',
        'Src IP', 'Dst IP', 'Src Port'  # 避免 overfitting
    ]

    # 自動建立分類資料資料夾
    os.makedirs(output_folder, exist_ok=True)

    # === 第一步：依類別分類儲存 ===
    for folder in input_folders:
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                print(f"處理檔案：{file}")
                path = os.path.join(folder, file)

                # 讀取資料
                df = dd.read_csv(path, assume_missing=True)
                df = df[[col for col in df.columns if col not in drop_columns]]

                # 轉換 Timestamp，自動判斷格式
                df[time_column] = dd.to_datetime(
                    df[time_column], format=time_format, errors="coerce")
                df = df.dropna(subset=[time_column, label_column])

                # 取得所有 label 類別
                labels = df[label_column].dropna().unique().compute().tolist()

                # 依類別分類儲存
                for label in labels:
                    print(label)
                    sub_df = df[df[label_column] == label]
                    label_filename = os.path.join(
                        output_folder, f"{label}.csv"
                    )

                    # 寫入或附加
                    if os.path.exists(label_filename):
                        sub_df.to_csv(label_filename, index=False,
                                      single_file=True, mode='a', header=False)
                    else:
                        sub_df.to_csv(label_filename, index=False,
                                      single_file=True)


def sample_benign():
    # 讀取原始 BENIGN 資料（安全記憶體方式）
    df = dd.read_csv("mess_data/BENIGN_30percent.csv")

    # 隨機抽樣保留 15% 資料
    df_sampled = df.sample(frac=0.5, random_state=520)

    # 輸出為新的 CSV
    df_sampled.to_csv("mess_data/BENIGN_15percent.csv",
                      index=False, single_file=True)
    print("✅ 已成功保留 15% BENIGN")


def merge_attempted():

    # === 路徑設定 ===
    benign_sampled_path = "mess_data/BENIGN_15percent.csv"
    attempted_pattern = "mess_data/*Attempted*.csv"
    output_path = "labeled_data/BENIGN_15_final.csv"

    # === 讀取欠採樣後的 BENIGN
    df_benign = dd.read_csv(benign_sampled_path)

    # === 讀取所有 Attempted 檔案並改 Label
    attempted_list = []
    for path in glob(attempted_pattern):
        print(f"📥 讀取 Attempted：{path}")
        df_attempted = dd.read_csv(path)
        df_attempted["Label"] = "BENIGN"
        attempted_list.append(df_attempted)

    # === 合併 BENIGN + 所有 Attempted
    df_all_benign = dd.concat([df_benign] + attempted_list)

    # === 輸出為單一 Parquet 檔案（更快更省記憶體）
    df_all_benign.to_csv(output_path, index=False, single_file=True)
    print("✅ 已成功產出整合後的 BENIGN + Attempted 資料！")


def sort_split_data():

    # === 參數設定 ===
    input_folder = "labeled_data"
    time_column = "Timestamp"
    label_column = "Label"
    time_format = "%Y-%m-%d %H:%M:%S.%f"
    train_ratio = 0.8

    # 結果暫存區
    train_list, val_list, label_stats = [], [], []

    # === 處理每個 CSV 檔案 ===
    for path in glob(f"{input_folder}/*.csv"):
        filename = os.path.basename(path)

        print(f"📂 處理：{filename}")
        label = filename.replace(".csv", "")

        if label == 'BENIGN_15_final':
            label = 'BENIGN'
            print(label)

        # 讀入資料（記憶體安全）
        ddf = dd.read_csv(path)

        # 時間轉換與清洗
        ddf[time_column] = dd.to_datetime(
            ddf[time_column], format=time_format, errors="coerce")
        ddf = ddf.dropna(subset=[time_column])
        ddf = ddf.sort_values(by=time_column)

        total_rows = len(ddf)
        print(total_rows)
        if total_rows < 2:
            print(f"⚠️ 跳過 {label}，資料太少")
            continue

        # 切分 80 / 20
        split_idx = int(total_rows * train_ratio)
        train_df = ddf.head(split_idx)
        val_df = ddf.tail(total_rows - split_idx)

        # 儲存
        train_list.append(train_df)
        val_list.append(val_df)

        # 統計分布
        label_stats.append({
            "Label": label,
            "Train Count": len(train_df),
            "Val Count": len(val_df),
            "Total": total_rows
        })

    # === 合併訓練集與驗證集 ===
    train_all = pd.concat(train_list).reset_index(drop=True)
    val_all = pd.concat(val_list).reset_index(drop=True)

    # 去除 Timestamp 欄位
    if time_column in train_all.columns:
        train_all = train_all.drop(columns=[time_column])
    if time_column in val_all.columns:
        val_all = val_all.drop(columns=[time_column])

    # === Label 編碼（將文字轉為整數）
    encoder = LabelEncoder()
    train_all[label_column] = encoder.fit_transform(train_all[label_column])
    val_all[label_column] = encoder.transform(val_all[label_column])

    # 儲存 label 對照表（轉換表）
    label_map_df = pd.DataFrame({
        "Label": encoder.classes_,
        "Encoded": range(len(encoder.classes_))
    })
    label_map_df.to_csv("label_encoding_map.csv", index=False)

    # === 輸出結果 ===
    train_all.to_csv("final_train.csv", index=False)
    val_all.to_csv("final_val.csv", index=False)
    pd.DataFrame(label_stats).to_csv("label_distribution.csv", index=False)

    print("✅ 完成！已產出：")
    print("  final_train.csv、final_val.csv（含編碼 Label）")
    print("  label_distribution.csv（每類筆數統計）")
    print("  label_encoding_map.csv（Label 編碼對照表）")


def get_csv_size():
    folder = "labeled_data"
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    total_size = 0
    file_sizes = []

    for f in csv_files:
        path = os.path.join(folder, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)  # bytes → MB
        total_size += size_mb
        file_sizes.append((f, round(size_mb, 2)))

    # 輸出每個檔案大小
    print("每個 Label 檔案大小（單位：MB）")
    for fname, size in sorted(file_sizes, key=lambda x: -x[1]):
        print(f"{fname:40s}  {size:7.2f} MB")

    # 輸出總大小
    print(f"總大小：{round(total_size, 2)} MB")


def get_demo(path):
    df = pd.read_csv(path, nrows=5)
    df.to_csv(f'{path}_demo.csv', index=False)
    print('demo saved')
    print(df)  # 預設印前 5 行


def clean_inf(input_path, output_path=None):
    """
    清理 inf，但保留 NaN 給 XGBoost 自動處理
    """
    print(f"📂 載入檔案：{input_path}")
    df = pd.read_csv(input_path)

    original_rows = len(df)

    # 強制轉數值，將 "inf" 字串等無效值轉成 NaN 或 inf
    df = df.apply(pd.to_numeric, errors='coerce')

    # 刪除含有 inf 的 row（保留 NaN）
    mask_inf = np.isinf(df).any(axis=1)
    cleaned_df = df[~mask_inf].copy()

    print(f"原始列數：{original_rows}")
    print(f"刪除 inf 列數：{mask_inf.sum()}")
    print(f"保留 NaN，清理後剩下：{len(cleaned_df)} 列")

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_cleaned.csv"

    cleaned_df.to_csv(output_path, index=False)
    print(f"✅ 輸出完成：{output_path}")


def check_data(df, name="Data"):
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.to_numpy()).sum()
    print(f"\n--- check {name} ---")
    print(f"numbers of NaN: {nan_count}")
    print(f"numbers of inf: {inf_count}")

def mutli2bin():
    path = 'final_data/final_train_cleaned.csv'
    df = pd.read_csv(path, low_memory=False)
    print('read')
    # 將 Label != 0 的全設為 1
    df["Label"] = df["Label"].apply(lambda x: 0 if x == 0 else 1)

    df.to_csv(f'{path}_binary.csv', index=False)
    df_demo = df.head()
    df_demo.to_csv(f'{path}_binary_demo.csv', index=False)


# get_csv_size()
# sample_benign()
# merge_attempted()
# sort_split_data()
# get_demo()
# clean_inf('final_train.csv', 'final_train_cleaned.csv')
# clean_inf('final_val.csv', 'final_val_cleaned.csv')
df = pd.read_csv("final_data/final_train_cleaned_binary.csv")
check_data(df)
df = pd.read_csv("final_data/final_val_cleaned_binary.csv")
check_data(df)
