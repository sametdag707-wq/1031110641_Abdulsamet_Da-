#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

from app_qt_mvp import read_audio_any, analyze_signal, Params, metrics_to_feature_dict

AUDIO_DIR = Path("data_audio")   # Ses dosyalarının klasörü
LABEL_CSV = Path("labels_reg.csv")
OUT_CSV   = Path("ovma_reg_dataset.csv")

def main():
    if not LABEL_CSV.exists():
        print(f"labels_reg.csv bulunamadı: {LABEL_CSV}")
        return

    labels = pd.read_csv(LABEL_CSV)
    labels["file_name"] = labels["file_name"].astype(str)

    rows = []

    for wav_path in AUDIO_DIR.rglob("*.*"):
        if wav_path.suffix.lower() not in [".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"]:
            continue

        fname = wav_path.name
        print("İşleniyor:", fname)

        row_label = labels[labels["file_name"] == fname]
        if row_label.empty:
            print("  -> labels_reg.csv'de karşılık yok, atlanıyor.")
            continue

        rpm = float(row_label["rpm"].iloc[0])
        ovma_score = float(row_label["ovma_score"].iloc[0])

        try:
            y, fs = read_audio_any(str(wav_path))
        except Exception as e:
            print("  -> Ses okunamadı:", e)
            continue

        p = Params(source="mic", audio_path=str(wav_path), rpm=rpm)
        metrics, _, _, _, _, _ = analyze_signal(y, fs, p, is_ae=False)

        feat = metrics_to_feature_dict(metrics)
        feat["ovma_score"] = ovma_score
        feat["rpm"] = rpm
        feat["file_name"] = fname

        rows.append(feat)

    if not rows:
        print("Hiç kayıt işlenmedi; rows listesi boş.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\nDataset kaydedildi →", OUT_CSV)
    print("Toplam kayıt:", len(df))

if __name__ == "__main__":
    main()
