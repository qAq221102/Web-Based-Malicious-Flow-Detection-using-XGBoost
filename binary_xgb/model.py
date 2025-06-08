from xgboost import XGBClassifier
import numpy as np


def build_binary_xgb():
    """ 
        建立一個二分類的 XGBoost 模型。
        回傳:
            XGBClassifier 物件
    """
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        missing=np.nan
    )
    return model
