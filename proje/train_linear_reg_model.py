#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

DATA_CSV  = Path("ovma_reg_dataset.csv")
MODEL_PKL = Path("ovma_linear_reg_model.pkl")

def main():
    if not DATA_CSV.exists():
        print(f"{DATA_CSV} bulunamadı.")
        return

    df = pd.read_csv(DATA_CSV)

    y = df["ovma_score"]
    drop_cols = ["ovma_score", "file_name"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression MAE:", mae)
    print("Linear Regression R² :", r2)

    joblib.dump(model, MODEL_PKL)
    print(f"Model kaydedildi → {MODEL_PKL}")

if __name__ == "__main__":
    main()
