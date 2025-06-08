from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.metrics import classification_report
from plot_util import plot_confusion_matrix
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# 配置
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
MODEL_PATH = "xgb_model.pkl"
ALLOWED_EXTENSIONS = {'csv'}

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 載入模型
model = joblib.load(MODEL_PATH)
print('✅ 模型載入成功')

# 必要特徵
REQUIRED_FEATURES = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd RST Flags', 'Bwd RST Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min',
    'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg',
    'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
    'Idle Max', 'Idle Min', 'ICMP Code', 'ICMP Type', 'Total TCP Flow Time'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_headers(headers):
    """檢查上傳的 CSV 是否包含所有必要的特徵，允許包含 Label 欄位"""
    headers = set(headers)
    if "Label" in headers:
        headers.remove("Label")
    missing = set(REQUIRED_FEATURES) - headers
    return len(missing) == 0, missing

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # 刪除檔案或捷徑
            elif os.path.isdir(file_path):
                # 遞迴刪除資料夾內容（如果有子資料夾）
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"無法刪除 {file_path}。錯誤：{e}")
            
def check_NaN_INF(df, name="Data"):
    nan_count = df.isna().sum().sum()
    inf_count = np.isinf(df.to_numpy()).sum()
    print(f"\n--- check {name} ---")
    print(f"numbers of NaN: {nan_count}")
    print(f"numbers of inf: {inf_count}")
    return nan_count,inf_count            
            
    
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        clear_folder('results')
        clear_folder('static')
        clear_folder('uploads')
        if 'file' not in request.files:
            flash('沒有選擇檔案','warning')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('沒有選擇檔案','warning')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('只接受 CSV 檔案','warning')
            return redirect(request.url)
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            df = pd.read_csv(filepath)
            headers_valid, missing_headers = check_headers(df.columns)
            print(headers_valid,missing_headers)
            if not headers_valid:
                os.remove(filepath)
                flash(f'缺少必要的特徵欄位: {", ".join(missing_headers)}','warning')
                return redirect(request.url)
            nan,inf=check_NaN_INF(df)
            if nan>0 or inf>0:
                flash(f'檔案中存在 NaN: {nan}個，數量太多時須留意','warning')
                flash(f'檔案中存在 INF: {inf}個','warning')
                os.remove(filepath)
                return redirect(request.url)
            
                
            y_true = df["Label"] if "Label" in df.columns else None
            X = df[REQUIRED_FEATURES]
            
            y_pred = model.predict(X)
            df["Prediction"] = y_pred

            result_filename = f"prediction_{filename}"
            result_path = os.path.join(RESULT_FOLDER, result_filename)
            df.to_csv(result_path, index=False)
            if y_true is not None :
                print(y_true)
                report = classification_report(y_true, y_pred, zero_division=0,digits=4)
                report_path = os.path.join(RESULT_FOLDER, f"report_{filename}.txt")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report)

                image_filename = f"confusion_matrix_{os.path.splitext(filename)[0]}"
                image_path = os.path.join("static", f'{image_filename}.png')
                plot_confusion_matrix(y_true, y_pred, target_dir="static",file_prefix=image_filename, normalize=False)
                flash('成功預測 & 產生混淆矩陣','success')
                return render_template("index.html",
                                       result_ready=True,
                                       result_path=f"/download/{result_filename}",
                                       report=report,
                                       image_path=image_path)
            
            flash('成功預測','success')
            return render_template("index.html",
                                   result_ready=True,
                                   result_path=f"/download/{result_filename}",
                                   report="無法產生分類報告（未提供真實標籤）")
        
        except Exception as e:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            flash(f'處理檔案時發生錯誤: {str(e)}','warning')
            print(e)
            return redirect(request.url)
    
    return render_template("index.html", result_ready=False)

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename),
                     as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
