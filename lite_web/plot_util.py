import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np


def plot_confusion_matrix(
    y_true,
    y_pred,
    target_dir="target",
    normalize=False,
    file_prefix="confusion_matrix",
    dpi=200
):
    """
    畫出並儲存中文混淆矩陣圖（支援標準版與百分比版）

    參數:
        y_true: 真實標籤（0=正常, 1=惡意）
        y_pred: 預測標籤（0=正常, 1=惡意）
        target_dir: 儲存資料夾
        normalize: 是否以百分比顯示（True 顯示百分比，False 顯示數字）
        file_prefix: 儲存檔案的前綴名稱
        dpi: 輸出解析度（適合網頁用）
    """

    labels = ["Benign", "Malicious"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 正規化成百分比（列方向）
    if normalize:
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot = np.array([["{0:.1f}%".format(x * 100)
                         for x in row] for row in cm_percent])
        fmt = ""
        title = "Confusion Matrix(percent)"
    else:
        annot = cm
        fmt = "d"
        title = "Confusion Matrix(value)"

    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor='gray')

    plt.title(title)
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    # 儲存 PNG / SVG 檔案
    png_path = os.path.join(target_dir, f"{file_prefix}.png")
    # svg_path = os.path.join(target_dir, f"{file_prefix}.svg")
    plt.savefig(png_path, dpi=dpi)
    # plt.savefig(svg_path, format='svg')
    plt.close()

    print(f"✅ confusion matrix is saved to: \n→ {png_path}")
    # print(f"✅ confusion matrix is saved to: \n→ {png_path}\n→ {svg_path}")
