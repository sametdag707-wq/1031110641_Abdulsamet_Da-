#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

DATA_CSV  = Path("ovma_reg_dataset.csv")
MODEL_PKL = Path("ovma_reg_model.pkl")

def main():
    if not DATA_CSV.exists():
        print(f"{DATA_CSV} bulunamadı, önce dataset oluşturun.")
        return

    df = pd.read_csv(DATA_CSV)

    if "ovma_score" not in df.columns:
        print("ovma_score kolonu bulunamadı.")
        return

    y = df["ovma_score"]
    drop_cols = ["ovma_score", "file_name"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("R² :", r2)

    joblib.dump(reg, MODEL_PKL)
    print(f"\nModel kaydedildi → {MODEL_PKL}")

if __name__ == "__main__":
    main()
