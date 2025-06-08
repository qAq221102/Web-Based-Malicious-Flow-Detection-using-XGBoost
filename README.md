# Web-Based Malicious Flow Detection using XGBoost

## Project Overview
This project is divided into two parts:
* binary_xgb/: for model training and data preprocessing
* lite_web/: for online prediction using a web interface

Because of file size limits on GitHub, the dataset used for training is provided through Google Drive:

https://drive.google.com/file/d/1LsDqBAd50fNVzVTj5EqKjxcZOG8b4YbM/view?usp=drive_link. 

Please follow the instructions below to download and use it.
1. Download and unzip
2. Move it into the binary_xgb folder. (you are supposed to see **binary_xgb/final_data/**)

We also provide:
* A project report in PDF format
* A demo video, which can be viewed using a QR code or the link below
<img src="https://github.com/user-attachments/assets/83232cf6-c5d3-4542-ab1e-a8eac2ffc6cd" alt="競賽影片QRcode" width="200">

(https://www.youtube.com/watch?v=WI7x1QNe3xQ)

## Poster
![作品海報](https://github.com/user-attachments/assets/5a83324d-4d53-4241-ac5d-04755fa76585)

## Abstract
本專案提出一套基於網頁介面的惡意流量二分類偵測系統。核心模型採用 XGBoost 二分類器，訓練資料來自提升版資料集 Improved CICIDS2017 & CSECICIDS2018。使用者可透過瀏覽器上傳已提取的網路封包特徵資料，系統將即時進行預測並提供結果下載。本系統以 Flask 架構實作，具備操作簡便、回應快速與部署彈性等特點，可應用於網路安全監控與入侵偵測等場域。

## Features
1. 創意
* 線上即時流量預測：
利用本團隊自行訓練並優化完成的 XGBoost 二分類模型，使用者只需透過網路上傳資料後能立刻就能下載正常或惡意的預測結果。
* 檔案格式檢查：
系統自動檢查是否為 CSV 檔並提示使用者轉換檔案格式。
* 自動特徵檢核：
系統自動檢測是否包含83個必要特徵並回報缺失的項目，簡化使用者的操作。
* 檔案內容檢查：
系統自動確認檔案內是否包含INF或NaN，並提示檔案內存在的數量。

2. 實用性
* 彈性 Label 偵測：
系統能自動偵測是否包含標籤資料，若無標籤，則僅輸出預測結果；反之，則生成混淆矩陣和分類報告。
* 提供下載預測結果、展示模型評估：
預測結果以 CSV 檔案供使用者下載，也會於網頁展示混淆矩陣、分類報告。
* 美觀、直覺的互動式介面：
乾淨利列的設計搭配動態生成元素，整個使用體驗不單一無趣；此外，一頁式的前端設計，降低了使用者進行流量預測的難度，無需專業知識即可輕鬆上手。

## Contributors

- Jiang Yi Sheng (@qAq221102)
- Wu Yu Ting (@qqqteresa)
