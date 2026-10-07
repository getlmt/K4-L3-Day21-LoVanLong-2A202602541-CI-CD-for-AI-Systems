import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Bonus 5: ty le lop duong tham chieu va muc lech toi da truoc khi canh bao drift
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05

# Bonus 2: cac nguong quyet dinh can quet (0.10 -> 0.90, buoc 0.05)
DECISION_THRESHOLDS = np.round(np.arange(0.10, 0.90 + 1e-9, 0.05), 2)


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # 1. Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # 2. Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: kiem tra lech phan phoi lop duong truoc khi huan luyen
    positive_rate = float(y_train.mean())
    if abs(positive_rate - REFERENCE_POSITIVE_RATE) > DRIFT_TOLERANCE:
        print(f"::warning::DATA DRIFT: ty le lop duong {positive_rate:.1%} lech qua "
              f"{DRIFT_TOLERANCE:.0%} so voi muc tham chieu {REFERENCE_POSITIVE_RATE:.1%}")
    else:
        print(f"Ty le lop duong: {positive_rate:.1%} (tham chieu {REFERENCE_POSITIVE_RATE:.1%}, khong drift)")

    with mlflow.start_run():

        # 3. Ghi nhan cac sieu tham so
        mlflow.log_params(params)
        mlflow.log_metric("positive_rate", positive_rate)

        # 4. Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 5. Du doan tren tap holdout va tinh chi so
        # f1_score tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1    = f1_score(y_eval, preds)
        acc   = accuracy_score(y_eval, preds)

        # 6. Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # Bonus 2: quet nguong quyet dinh tren xac suat lop duong
        proba = model.predict_proba(X_eval)[:, 1]
        f1_by_threshold = {float(t): f1_score(y_eval, (proba >= t).astype(int)) for t in DECISION_THRESHOLDS}
        best_threshold = max(f1_by_threshold, key=f1_by_threshold.get)
        best_f1 = f1_by_threshold[best_threshold]
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("f1_best_threshold", best_f1)

        # 7. In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Nguong toi uu: {best_threshold:.2f} -> F1 {best_f1:.4f} (nguong mac dinh 0.50 -> F1 {f1:.4f})")

        # 8. Luu metrics ra file outputs/report.json (GitHub Actions doc o Buoc 2)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({
                "f1_score": f1,
                "accuracy": acc,
                "positive_rate": positive_rate,
                "best_threshold": best_threshold,
                "f1_best_threshold": best_f1,
            }, f)

        # 9. Luu mo hinh ra file models/model.joblib (upload len cloud storage o Buoc 2)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # 10. Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
