#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qt AE/Mic Analysis MVP (Rev-3, canlı dinleme + referans + ML Ovma Skoru)
- PySide6 + Matplotlib gömülü grafikler
- Kaynak: Mic/Telefon (WAV/FLAC/OGG/MP3/M4A/AAC) veya AE (.pridb+.tradb)
- Grafikler: Zaman, PSD, Zarf Tayfı, Spektrogram, Trend (RMS/Crest)
- Canlı dinleme (mikrofon), referans eğri, canlı uyarı
- Faz-2: ML ovma skoru (0–100) entegrasyonu + ovma etiketi kaydetme

Bağımlılıklar (venv içinde):
  python -m pip install PySide6 matplotlib numpy scipy soundfile pandas
  python -m pip install librosa           # M4A/MP3 için (opsiyonel)
  python -m pip install vallenae         # AE dosyaları için (opsiyonel)
  python -m pip install sounddevice      # canlı dinleme için (opsiyonel)
  python -m pip install scikit-learn joblib

Çalıştırma:
  python app_qt_mvp.py
"""
from __future__ import annotations

import os
import sys
import json
import math
import warnings
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import signal

# Matplotlib (Qt backend)
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

# Audio IO (mic/phone)
import soundfile as sf
try:
    import librosa  # m4a/ffmpeg fallback
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

# AE reader (optional)
try:
    import vallenae as vae
    VALLENAE_AVAILABLE = True
except Exception:
    VALLENAE_AVAILABLE = False

# Canlı ses (mikrofon)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

# ML model yükleme
import joblib

# Qt
from PySide6 import QtCore, QtWidgets, QtGui


# ------------------------- Config / Defaults ------------------------- #
MIC_BANDS_HZ = [(100.0, 500.0), (500.0, 2000.0), (2000.0, 8000.0)]
AE_BANDS_HZ = [(100e3, 300e3), (300e3, 600e3), (600e3, 800e3)]
AE_ENV_BAND = (100e3, 300e3)
MIC_ENV_BAND_MARGIN = 0.15  # env band: [TPF*(1-m), TPF*(1+m)]


@dataclass
class Params:
    source: str = "mic"  # "mic" | "ae"
    audio_path: Optional[str] = None
    pridb_path: Optional[str] = None
    tradb_path: Optional[str] = None

    rpm: Optional[float] = None
    teeth: Optional[int] = None
    tool_diam_mm: Optional[float] = None
    material: Optional[str] = None
    E_GPa: Optional[float] = None
    G_GPa: Optional[float] = None
    feed_mm_min: Optional[float] = None
    ap_mm: Optional[float] = None
    ae_mm: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class Metrics:
    fs_Hz: float
    rms_V: float
    peak_V: float
    crest: float
    kurtosis: float
    f_peak_hz: float
    bw_3db_hz: float
    band_powers: dict
    env_mod_peak_hz: float
    env_rms: float
    env_peak: float
    tpf_hz: Optional[float]
    R_h_dB: Optional[dict]
    risk_score: float
    risk_label: str


def metrics_to_feature_dict(m: Metrics) -> dict:
    """
    Metrics dataclass'ini ML için tek satırlık feature dict'e çevirir.
    Ovma modeli hem bu özellikleri hem de (varsa) rpm bilgisini kullanacak.
    """
    d = {
        "fs_Hz": m.fs_Hz,
        "rms_V": m.rms_V,
        "peak_V": m.peak_V,
        "crest": m.crest,
        "kurtosis": m.kurtosis,
        "f_peak_hz": m.f_peak_hz,
        "bw_3db_hz": m.bw_3db_hz,
        "env_mod_peak_hz": m.env_mod_peak_hz,
        "env_rms": m.env_rms,
        "env_peak": m.env_peak,
        "tpf_hz": m.tpf_hz if m.tpf_hz is not None else 0.0,
        "band1": m.band_powers.get("band1", 0.0),
        "band2": m.band_powers.get("band2", 0.0),
        "band3": m.band_powers.get("band3", 0.0),
    }

    # TPF yan bant oranları (h1..h5)
    if m.R_h_dB:
        for k, v in m.R_h_dB.items():
            key = f"R_{k}"
            d[key] = float(v) if np.isfinite(v) else 0.0
    else:
        for h in range(1, 6):
            d[f"R_h{h}"] = 0.0

    return d


# ------------------------- Processing helpers ----------------------- #

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(float)
    return np.mean(x, axis=1).astype(float)

def read_with_ffmpeg_cli(path: str) -> Tuple[np.ndarray, float]:
    """
    ffmpeg komut satırı ile dosyayı geçici bir WAV'a çevir,
    sonra soundfile ile oku.
    Burada ffmpeg'in stderr çıkışını da yakalayıp anlamlı hata mesajı üretiyoruz.
    """
    in_path = os.path.normpath(path)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    tmp_path = os.path.normpath(tmp_path)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", in_path,
            "-ac", "1",
            tmp_path,
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            err_msg = (proc.stderr or "").strip()
            if not err_msg:
                err_msg = "ffmpeg bilinmeyen hata (stderr boş)"
            raise RuntimeError(f"ffmpeg hata (exit {proc.returncode}): {err_msg}")

        y, fs = sf.read(tmp_path)
        y = to_mono(y)
        if np.max(np.abs(y)) > 1.5:
            y = y / np.max(np.abs(y))
        return y.astype(float), float(fs)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def read_audio_any(path: str) -> Tuple[np.ndarray, float]:
    """Ses dosyasını mümkün olduğunca esnek okuyucu."""
    errors = []

    # 1) soundfile
    try:
        y, fs = sf.read(path)
        y = to_mono(y)
        if np.max(np.abs(y)) > 1.5:
            y = y / np.max(np.abs(y))
        return y.astype(float), float(fs)
    except Exception as e_sf:
        errors.append(f"soundfile hata: {type(e_sf).__name__}: {e_sf}")

    # 2) librosa
    if LIBROSA_AVAILABLE:
        try:
            print(f"[read_audio_any] librosa ile okunuyor: {path}")
            y, fs = librosa.load(path, sr=None, mono=True)
            if np.max(np.abs(y)) > 1.5:
                y = y / np.max(np.abs(y))
            return y.astype(float), float(fs)
        except Exception as e_lib:
            errors.append(f"librosa hata: {type(e_lib).__name__}: {e_lib}")
    else:
        errors.append("librosa yüklü değil (pip install librosa).")

    # 3) ffmpeg CLI
    try:
        print(f"[read_audio_any] ffmpeg CLI ile deneniyor: {path}")
        y, fs = read_with_ffmpeg_cli(path)
        return y.astype(float), float(fs)
    except Exception as e_ff:
        errors.append(f"ffmpeg CLI hata: {type(e_ff).__name__}: {e_ff}")

    msg = "Ses dosyası okunamadı.\n" + "\n".join(errors)
    msg += (
        "\n\n• soundfile tipik olarak WAV/FLAC/OGG formatlarını destekler.\n"
        "• librosa + audioread, FFmpeg backend ile M4A/MP3 açmayı dener.\n"
        "• ffmpeg CLI fallback bile bu dosyayı açamadıysa, dosya bozuk olabilir veya "
        "çok alışılmadık bir kodek kullanıyor olabilir."
    )
    raise RuntimeError(msg)


def pick_nperseg(n: int) -> int:
    base = max(2048, 2 ** int(np.floor(np.log2(max(1024, n // 8)))))
    return int(min(base, 262_144))


def band_power(f: np.ndarray, Pxx: np.ndarray, lo: float, hi: float) -> float:
    if hi <= lo:
        return float("nan")
    idx = (f >= lo) & (f <= hi)
    if not np.any(idx):
        return float("nan")
    return float(np.trapz(Pxx[idx], f[idx]))


def psd_welch(y: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    nperseg = pick_nperseg(len(y))
    f, Pxx = signal.welch(
        y, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
        detrend="constant", scaling="density"
    )
    i_pk = int(np.argmax(Pxx))
    f_pk = float(f[i_pk])
    p_pk = float(Pxx[i_pk])
    idx_3db = Pxx >= (p_pk / 2.0)
    bw_3db = float(f[idx_3db][-1] - f[idx_3db][0]) if np.any(idx_3db) else float("nan")
    return f, Pxx, f_pk, p_pk, bw_3db


def envelope_spectrum(y: np.ndarray, fs: float, band: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    nyq = fs / 2.0
    lo, hi = band
    if hi >= nyq:
        hi = nyq * 0.98
    if lo <= 0:
        lo = 1.0
    b, a = signal.butter(4, [lo / nyq, hi / nyq], btype="band")
    xb = signal.filtfilt(b, a, y)
    env = np.abs(signal.hilbert(xb))
    nperseg = max(2048, int(fs // 50))
    fe, Pe = signal.welch(
        env - np.mean(env),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=0,
        scaling="density",
    )
    idx = fe >= 1.0
    fe2 = fe[idx]
    Pe2 = Pe[idx]
    if len(Pe2):
        i_pk = int(np.argmax(Pe2))
        f_pk = float(fe2[i_pk])
        env_rms = float(np.sqrt(np.mean(env ** 2)))
        env_peak = float(np.max(env))
    else:
        f_pk, env_rms, env_peak = 0.0, 0.0, 0.0
    return fe, Pe, f_pk, env_rms, env_peak


def tpf_metrics(
    f: np.ndarray,
    Pxx: np.ndarray,
    rpm: Optional[float],
    teeth: Optional[int],
) -> Tuple[Optional[dict], Optional[float]]:
    if not rpm or not teeth:
        return None, None
    f_rev = rpm / 60.0
    f_tpf = teeth * f_rev
    H = 5
    ratios = {}
    for h in range(1, H + 1):
        f0 = h * f_tpf
        d = max(10.0, 0.02 * f0)  # center half-width
        W = max(100.0, 0.3 * f0)  # ring half-width
        Pc = band_power(f, Pxx, f0 - d, f0 + d)
        Pr = band_power(f, Pxx, f0 - W, f0 - d) + band_power(f, Pxx, f0 + d, f0 + W)
        if Pc <= 0 or not np.isfinite(Pc):
            ratios[f"h{h}"] = float("nan")
        else:
            ratios[f"h{h}"] = 10.0 * math.log10(max(Pr, 1e-30) / Pc)
    return ratios, f_tpf


def compute_trend(y: np.ndarray, fs: float, win_s: float = 0.05, step_s: Optional[float] = None):
    if step_s is None:
        step_s = win_s / 2.0
    n = len(y)
    w = max(8, int(win_s * fs))
    s = max(1, int(step_s * fs))
    t0 = []
    rms = []
    crest = []
    for start in range(0, n - w + 1, s):
        seg = y[start:start + w]
        r = np.sqrt(np.mean(seg ** 2))
        p = np.max(np.abs(seg)) if len(seg) else 0.0
        t0.append((start + w / 2) / fs)
        rms.append(r)
        crest.append(p / (r + 1e-12))
    return np.asarray(t0), np.asarray(rms), np.asarray(crest)


def spectrogram_db(y: np.ndarray, fs: float):
    nperseg = pick_nperseg(len(y))
    nover = nperseg // 2
    f, t, Sxx = signal.spectrogram(
        y,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nover,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    Sxx_dB = 10.0 * np.log10(Sxx + 1e-24)
    return f, t, Sxx_dB


def risk_from_metrics(m: Metrics, is_ae: bool) -> Tuple[float, str]:
    # Normalize few core metrics to 0..1 roughly (heuristic)
    norm_crest = min(m.crest / 5.0, 2.0) / 2.0
    norm_kurt = min(m.kurtosis / 10.0, 2.0) / 2.0
    # choose high band according to mode
    hi_band = "band3"
    bp = 0.0
    if hi_band in m.band_powers:
        bp = m.band_powers[hi_band]
    norm_bp = min(1.0, np.log10(bp + 1e-18) + 18.0) / 6.0
    norm_r = 0.0
    if m.R_h_dB:
        vals = [v for v in m.R_h_dB.values() if np.isfinite(v)]
        if vals:
            norm_r = min(max((max(vals) + 10.0) / 20.0, 0.0), 1.0)
    score = 25 * norm_crest + 25 * norm_kurt + 25 * norm_bp + 25 * norm_r
    label = "OK"
    if score >= 65:
        label = "Uyarı"
    elif score >= 35:
        label = "Dikkat"
    return score, label


def analyze_signal(y: np.ndarray, fs: float, params: Params, is_ae: bool):
    y = to_mono(y)
    # basic stats
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    crest = float(peak / (rms + 1e-12))
    kurt = float(np.mean((y - np.mean(y)) ** 4) / (np.var(y) ** 2 + 1e-18))

    # PSD
    f, Pxx, f_pk, p_pk, bw_3db = psd_welch(y, fs)

    # Band powers
    bands = AE_BANDS_HZ if is_ae else MIC_BANDS_HZ
    band_powers = {}
    for i, (lo, hi) in enumerate(bands, start=1):
        band_powers[f"band{i}"] = band_power(f, Pxx, lo, hi)

    # Envelope spectrum
    if is_ae:
        env_band = AE_ENV_BAND
    else:
        if params.rpm and params.teeth:
            f_tpf_guess = (params.rpm / 60.0) * params.teeth
            env_band = (
                max(10.0, f_tpf_guess * (1.0 - MIC_ENV_BAND_MARGIN)),
                f_tpf_guess * (1.0 + MIC_ENV_BAND_MARGIN),
            )
        else:
            env_band = (200.0, 1200.0)
    fe, Pe, env_pk, env_rms, env_peak = envelope_spectrum(y, fs, env_band)

    # TPF sideband ratios
    ratios, f_tpf = tpf_metrics(f, Pxx, params.rpm, params.teeth)

    # Risk
    tmp = Metrics(
        fs_Hz=fs,
        rms_V=rms,
        peak_V=peak,
        crest=crest,
        kurtosis=kurt,
        f_peak_hz=f_pk,
        bw_3db_hz=bw_3db,
        band_powers=band_powers,
        env_mod_peak_hz=env_pk,
        env_rms=env_rms,
        env_peak=env_peak,
        tpf_hz=f_tpf,
        R_h_dB=ratios,
        risk_score=0.0,
        risk_label="",
    )
    score, label = risk_from_metrics(tmp, is_ae)
    tmp.risk_score, tmp.risk_label = score, label

    # Extras: trend + spectrogram + time signal
    t_sig = np.arange(len(y)) / fs
    t_tr, tr_rms, tr_crest = compute_trend(y, fs)
    fspec, tspec, S_dB = spectrogram_db(y, fs)

    return tmp, (f, Pxx), (fe, Pe), (t_sig, y), (tspec, fspec, S_dB), (t_tr, tr_rms, tr_crest)


# ------------------------- Matplotlib Panels ------------------------ #
class MplPanel(QtWidgets.QWidget):
    def __init__(self, title: str, info_text: str = "", parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavToolbar(self.canvas, self)
        self.info_text = info_text
        self.info_btn = QtWidgets.QPushButton("i")
        self.info_btn.setFixedWidth(26)
        self.info_btn.setToolTip("Grafik rehberi")
        self.info_btn.clicked.connect(self.show_info)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.toolbar)
        top.addStretch(1)
        top.addWidget(self.info_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title)

    def show_info(self):
        txt = self.info_text or "Bu grafik için rehber daha sonra eklenecek."
        QtWidgets.QMessageBox.information(self, "Grafik Rehberi", txt)

    def plot_wave(self, t: np.ndarray, y: np.ndarray, ref: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.ax.clear()
        if ref is not None:
            t_ref, y_ref = ref
            self.ax.plot(t_ref * 1e6, y_ref * 1e3, color="0.8", linewidth=1.0, alpha=0.7, label="Ref")
        self.ax.plot(t * 1e6, y * 1e3, label="Sinyal")
        self.ax.set_xlabel("Zaman [µs]")
        self.ax.set_ylabel("Genlik [mV]")
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()

    def plot_psd(
        self,
        f: np.ndarray,
        Pxx: np.ndarray,
        tpf: Optional[float] = None,
        ref: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.ax.clear()
        if ref is not None:
            f_ref, Pxx_ref = ref
            self.ax.loglog(f_ref, Pxx_ref + 1e-24, color="0.8", linewidth=1.0, alpha=0.7, label="Ref")
        self.ax.loglog(f, Pxx + 1e-24, label="Sinyal")
        if tpf and np.isfinite(tpf):
            for h in range(1, 6):
                x = h * tpf
                self.ax.axvline(x, linestyle=":", linewidth=1, alpha=0.7)
        self.ax.set_xlabel("Frekans [Hz] (log)")
        self.ax.set_ylabel("PSD [V²/Hz] (log)")
        self.ax.grid(True, which="both", alpha=0.2)
        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()

    def plot_env(
        self,
        fe: np.ndarray,
        Pe: np.ndarray,
        ref: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.ax.clear()
        if ref is not None:
            fe_ref, Pe_ref = ref
            self.ax.plot(fe_ref, Pe_ref, color="0.8", linewidth=1.0, alpha=0.7, label="Ref")
        self.ax.plot(fe, Pe, label="Sinyal")
        self.ax.set_xlim(0, max(20000.0, float(fe[-1]) if len(fe) else 20000.0))
        self.ax.set_xlabel("Frekans [Hz]")
        self.ax.set_ylabel("Zarf PSD [a.u.]")
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()


class SpecPanel(MplPanel):
    def __init__(self, title: str, info_text: str = "", parent=None):
        super().__init__(title, info_text, parent)
        self.im = None
        self.cbar = None
        self.figure.subplots_adjust(right=0.88)

    def plot_spec(self, t: np.ndarray, f: np.ndarray, S_dB: np.ndarray):
        if t is None or f is None or S_dB is None or len(t) == 0 or len(f) == 0:
            self.ax.clear()
            self.ax.set_xlabel("Zaman [s]")
            self.ax.set_ylabel("Frekans [Hz]")
            self.canvas.draw_idle()
            return

        self.ax.clear()

        T, F = np.meshgrid(t, f)
        self.im = self.ax.pcolormesh(T, F, S_dB, shading="auto")

        self.ax.set_xlabel("Zaman [s]")
        self.ax.set_ylabel("Frekans [Hz]")
        self.ax.set_yscale("log")
        self.ax.set_ylim(max(1.0, float(f[0])), float(f[-1]))

        if self.cbar is None:
            self.cbar = self.figure.colorbar(self.im, ax=self.ax, pad=0.02)
            self.cbar.set_label("PSD [dB]")
        else:
            self.cbar.mappable = self.im
            self.cbar.update_normal(self.im)

        self.canvas.draw_idle()


class TrendPanel(MplPanel):
    def plot_trend(
        self,
        t: np.ndarray,
        rms: np.ndarray,
        crest: np.ndarray,
        ref: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        self.ax.clear()
        if ref is not None:
            t_ref, rms_ref, crest_ref = ref
            self.ax.plot(t_ref, rms_ref, color="0.8", linewidth=1.0, alpha=0.7, label="RMS Ref")
            self.ax.plot(t_ref, crest_ref, color="0.6", linewidth=1.0, alpha=0.7, label="Crest Ref")
        self.ax.plot(t, rms, label="RMS")
        self.ax.plot(t, crest, label="Crest")
        self.ax.set_xlabel("Zaman [s]")
        self.ax.set_ylabel("Değer")
        self.ax.legend()
        self.ax.grid(True, alpha=0.2)
        self.canvas.draw_idle()


# ------------------------- Main Window ------------------------------ #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ses / AE Analiz MVP + ML Ovma Skoru")
        self.resize(1380, 820)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        # Left panel: params form
        self.form_group = QtWidgets.QGroupBox("Parametreler")
        self.form_layout = QtWidgets.QFormLayout(self.form_group)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Mic/Telefon", "AE (Vallen)"])
        self.form_layout.addRow("Kaynak", self.source_combo)

        # File pickers (stacked)
        self.stack_files = QtWidgets.QStackedWidget()
        # Mic widget
        mic_w = QtWidgets.QWidget()
        mic_l = QtWidgets.QHBoxLayout(mic_w)
        self.ed_audio = QtWidgets.QLineEdit()
        self.btn_audio = QtWidgets.QPushButton("Dosya Seç")
        mic_l.addWidget(self.ed_audio)
        mic_l.addWidget(self.btn_audio)
        self.stack_files.addWidget(mic_w)
        # AE widget
        ae_w = QtWidgets.QWidget()
        ae_form = QtWidgets.QFormLayout(ae_w)
        self.ed_pridb = QtWidgets.QLineEdit()
        self.btn_pridb = QtWidgets.QPushButton(".pridb")
        self.ed_tradb = QtWidgets.QLineEdit()
        self.btn_tradb = QtWidgets.QPushButton(".tradb")
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(self.ed_pridb)
        row1.addWidget(self.btn_pridb)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(self.ed_tradb)
        row2.addWidget(self.btn_tradb)
        ae_form.addRow("PriDB", row1)
        ae_form.addRow("TraDB", row2)
        # TRAI selector
        self.combo_trai = QtWidgets.QComboBox()
        self.combo_trai.setEnabled(False)
        ae_form.addRow("TRAI", self.combo_trai)
        self.stack_files.addWidget(ae_w)

        self.form_layout.addRow("Dosyalar", self.stack_files)

        # Numeric fields
        self.spin_rpm = QtWidgets.QDoubleSpinBox()
        self.spin_rpm.setRange(0, 1e6)
        self.spin_rpm.setDecimals(1)
        self.spin_teeth = QtWidgets.QSpinBox()
        self.spin_teeth.setRange(0, 1000)
        self.spin_tool = QtWidgets.QDoubleSpinBox()
        self.spin_tool.setRange(0, 1000)
        self.spin_tool.setDecimals(3)
        self.ed_material = QtWidgets.QLineEdit()
        self.spin_E = QtWidgets.QDoubleSpinBox()
        self.spin_E.setRange(0, 1e4)
        self.spin_E.setDecimals(2)
        self.spin_G = QtWidgets.QDoubleSpinBox()
        self.spin_G.setRange(0, 1e4)
        self.spin_G.setDecimals(2)
        self.spin_feed = QtWidgets.QDoubleSpinBox()
        self.spin_feed.setRange(0, 1e6)
        self.spin_feed.setDecimals(2)
        self.spin_ap = QtWidgets.QDoubleSpinBox()
        self.spin_ap.setRange(0, 1000)
        self.spin_ap.setDecimals(3)
        self.spin_ae = QtWidgets.QDoubleSpinBox()
        self.spin_ae.setRange(0, 1000)
        self.spin_ae.setDecimals(3)
        self.txt_notes = QtWidgets.QTextEdit()
        self.txt_notes.setFixedHeight(60)

        self.form_layout.addRow("RPM", self.spin_rpm)
        self.form_layout.addRow("Diş Sayısı", self.spin_teeth)
        self.form_layout.addRow("Takım Çapı [mm]", self.spin_tool)
        self.form_layout.addRow("Malzeme", self.ed_material)
        self.form_layout.addRow("E [GPa]", self.spin_E)
        self.form_layout.addRow("G [GPa]", self.spin_G)
        self.form_layout.addRow("İlerleme [mm/min]", self.spin_feed)
        self.form_layout.addRow("ap [mm]", self.spin_ap)
        self.form_layout.addRow("ae [mm]", self.spin_ae)
        self.form_layout.addRow("Not", self.txt_notes)

        # Ovma etiketi (manuel label girişi)
        self.spin_ovma_label = QtWidgets.QSpinBox()
        self.spin_ovma_label.setRange(0, 100)
        self.spin_ovma_label.setValue(0)
        self.spin_ovma_label.setSuffix(" /100")
        self.btn_save_ovma_label = QtWidgets.QPushButton("Ovma Skorunu Kaydet")
        self.form_layout.addRow("Ovma Skoru (etiket)", self.spin_ovma_label)
        self.form_layout.addRow(self.btn_save_ovma_label)

        # Buttons
        self.btn_analyze = QtWidgets.QPushButton("Analiz Et")
        self.btn_save = QtWidgets.QPushButton("Çıktıları Kaydet")
        self.btn_live = QtWidgets.QPushButton("Canlı Dinle")
        self.btn_stop_live = QtWidgets.QPushButton("Durdur")
        self.btn_set_ref = QtWidgets.QPushButton("Referans Yap")
        self.btn_play = QtWidgets.QPushButton("Ses Çal")

        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addWidget(self.btn_analyze)
        btn_box.addWidget(self.btn_save)
        btn_box.addWidget(self.btn_live)
        btn_box.addWidget(self.btn_stop_live)
        btn_box.addWidget(self.btn_set_ref)
        self.form_layout.addRow(btn_box)
        btn_box.addWidget(self.btn_play)

        hbox.addWidget(self.form_group, 0)

        # Right side: tabs with plots + summary
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)

        # Info texts per panel
        info_time = (
            "ZAMAN DALGASI\n\n"
            "Ham AE/ses dalga formu.\n"
            "• Zirve, RMS, crest ve sönümü gözle.\n"
            "• Uzayan ring-down ve yüksek crest → darbeli/rezonanslı olay."
        )
        info_psd = (
            "PSD (Güç Spektral Yoğunluğu)\n\n"
            "Enerjinin frekansa dağılımı.\n"
            "• Tepe frekansı ve -3 dB bant genişliği rezonansları gösterir.\n"
            "• TPF ve harmoniklerinde dikey çizgiler: tırlama için yan bantları izle."
        )
        info_env = (
            "ZARF TAYFI\n\n"
            "Yüksek frekans taşıyıcının zarfının spektrumu.\n"
            "• Periyodik darbeler varsa kHz bölgesinde tepe oluşur.\n"
            "• Rulman/diş geçişi/çatter modülasyonlarını ortaya çıkarır."
        )
        info_spec = (
            "SPEKTROGRAM\n\n"
            "Zaman–frekans ısı haritası (dB).\n"
            "• Bantların zamanla nasıl aktive olduğunu gösterir.\n"
            "• Geçici olayları, TPF çevresindeki şeritleri gör."
        )
        info_trend = (
            "TREND (RMS & CREST)\n\n"
            "Kayan pencerede RMS ve crest.\n"
            "• RMS yavaş artışı: ovma/aşınma.\n"
            "• Crest sıçramaları: darbeli olaylar."
        )

        self.tabs = QtWidgets.QTabWidget()
        self.tab_wave = MplPanel("Zaman", info_time)
        self.tab_psd = MplPanel("PSD", info_psd)
        self.tab_env = MplPanel("Zarf", info_env)
        self.tab_spec = SpecPanel("Spektrogram", info_spec)
        self.tab_trend = TrendPanel("Trend", info_trend)
        self.tabs.addTab(self.tab_wave, "Zaman")
        self.tabs.addTab(self.tab_psd, "PSD")
        self.tabs.addTab(self.tab_env, "Zarf")
        self.tabs.addTab(self.tab_spec, "Spektrogram")
        self.tabs.addTab(self.tab_trend, "Trend")

        rv.addWidget(self.tabs, 1)

        # Summary box
        self.grp_summary = QtWidgets.QGroupBox("Otomatik Yorum & Risk")
        gl = QtWidgets.QGridLayout(self.grp_summary)
        self.lbl_risk = QtWidgets.QLabel("Risk: -")
        self.lbl_risk.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_risk.setStyleSheet(
            "QLabel {font-weight:600; padding:6px; border-radius:8px; background:#ddd}"
        )
        self.txt_summary = QtWidgets.QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMinimumHeight(160)
        gl.addWidget(self.lbl_risk, 0, 0, 1, 1)
        gl.addWidget(self.txt_summary, 1, 0, 1, 1)

        rv.addWidget(self.grp_summary, 0)
        hbox.addWidget(right, 1)

        # Connections
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        self.btn_audio.clicked.connect(self.pick_audio)
        self.btn_pridb.clicked.connect(self.pick_pridb)
        self.btn_tradb.clicked.connect(self.pick_tradb)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_save.clicked.connect(self.save_outputs)
        self.btn_set_ref.clicked.connect(self.set_reference_from_current)
        self.btn_live.clicked.connect(self.start_live)
        self.btn_stop_live.clicked.connect(self.stop_live)
        self.combo_trai.currentIndexChanged.connect(self.on_trai_change)
        self.btn_play.clicked.connect(self.play_current_audio)
        self.btn_save_ovma_label.clicked.connect(self.save_ovma_label)

        # Initial state
        self.on_source_changed(0)
        self.current_signal: Optional[Tuple[np.ndarray, float]] = None
        self.current_metrics: Optional[Metrics] = None
        self.spec_cache = None
        self.trend_cache = None
        self.last_psd = None
        self.last_env = None

        # Referans (baseline)
        self.ref_metrics: Optional[Metrics] = None
        self.ref_wave = None
        self.ref_psd = None
        self.ref_env = None
        self.ref_trend = None

        # Canlı dinleme
        self.live_stream = None
        self.live_timer: Optional[QtCore.QTimer] = None
        self.live_buffer: Optional[np.ndarray] = None
        self.live_write_idx: int = 0
        self.live_fs: float = 48000.0
        self.live_win_sec: float = 2.0
        self.is_playing_audio: bool = False

        # ML ovma skoru modeli
        self.ovma_model = None
        self.ovma_feature_cols = None
        self.load_ovma_model()

    # ----------------- UI helpers -----------------
    def on_source_changed(self, idx: int):
        if idx == 0:  # mic
            self.stack_files.setCurrentIndex(0)
        else:
            self.stack_files.setCurrentIndex(1)
            if not VALLENAE_AVAILABLE:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Vallenae yok",
                    "AE modu için 'vallenae' paketini kurun: pip install vallenae",
                )

    def pick_audio(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Ses Dosyası Seç",
            "",
            "Audio (*.wav *.flac *.ogg *.mp3 *.m4a *.aac);;All files (*.*)",
        )
        if fn:
            self.ed_audio.setText(fn)

    def pick_pridb(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, ".pridb seç", "", "PriDB (*.pridb)"
        )
        if fn:
            self.ed_pridb.setText(fn)
            self.try_fill_trai()

    def pick_tradb(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, ".tradb seç", "", "TraDB (*.tradb)"
        )
        if fn:
            self.ed_tradb.setText(fn)
            self.try_fill_trai()

    def try_fill_trai(self):
        if not VALLENAE_AVAILABLE:
            return
        pr, tr = self.ed_pridb.text().strip(), self.ed_tradb.text().strip()
        if pr and tr and Path(pr).exists() and Path(tr).exists():
            try:
                with vae.io.PriDatabase(Path(pr)) as db:
                    df = db.read_hits()
                if "trai" in df.columns:
                    self.combo_trai.blockSignals(True)
                    self.combo_trai.clear()
                    for v in df["trai"].tolist():
                        self.combo_trai.addItem(str(int(v)))
                    self.combo_trai.setEnabled(True)
                    self.combo_trai.blockSignals(False)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "TRAI okunamadı", str(e))

    def on_trai_change(self, _idx: int):
        pass

    def gather_params(self) -> Params:
        p = Params()
        p.source = "mic" if self.source_combo.currentIndex() == 0 else "ae"
        p.audio_path = self.ed_audio.text().strip() or None
        p.pridb_path = self.ed_pridb.text().strip() or None
        p.tradb_path = self.ed_tradb.text().strip() or None
        p.rpm = self.spin_rpm.value() or None
        p.teeth = int(self.spin_teeth.value()) or None
        p.tool_diam_mm = self.spin_tool.value() or None
        p.material = self.ed_material.text().strip() or None
        p.E_GPa = self.spin_E.value() or None
        p.G_GPa = self.spin_G.value() or None
        p.feed_mm_min = self.spin_feed.value() or None
        p.ap_mm = self.spin_ap.value() or None
        p.ae_mm = self.spin_ae.value() or None
        p.notes = self.txt_notes.toPlainText().strip() or None
        return p

    # ----------------- Offline Analysis -----------------
    def run_analysis(self):
        import traceback

        p = self.gather_params()
        try:
            print("=== run_analysis çağrıldı ===")
            print("Kaynak:", p.source)
            print("Audio path:", p.audio_path)

            if p.source == "mic":
                if not p.audio_path:
                    raise RuntimeError("Ses dosyası seçilmedi.")
                if not Path(p.audio_path).exists():
                    raise RuntimeError(f"Ses dosyası bulunamadı:\n{p.audio_path}")
                y, fs = read_audio_any(p.audio_path)
            else:  # AE modu
                if not VALLENAE_AVAILABLE:
                    raise RuntimeError("AE modu için 'vallenae' gerekli.")
                if not (p.pridb_path and p.tradb_path):
                    raise RuntimeError(".pridb ve .tradb dosyalarını seçiniz.")
                if not (Path(p.pridb_path).exists() and Path(p.tradb_path).exists()):
                    raise RuntimeError("AE dosya yolları bulunamadı.")

                if self.combo_trai.count() > 0:
                    trai = int(self.combo_trai.currentText())
                else:
                    with vae.io.PriDatabase(Path(p.pridb_path)) as db:
                        df = db.read_hits()
                        trai = int(df["trai"].iloc[0])

                with vae.io.TraDatabase(Path(p.tradb_path)) as tradb:
                    y, t = tradb.read_wave(trai=trai, time_axis=True, raw=False)
                    fs = 1.0 / float(np.mean(np.diff(t)))

            print("Sinyal uzunluğu:", len(y), "fs:", fs)
            if len(y) == 0:
                raise RuntimeError("Okunan sinyal boş (length=0).")

            res = analyze_signal(y, fs, p, is_ae=(p.source == "ae"))
            (
                metrics,
                (f, Pxx),
                (fe, Pe),
                (t_sig, yw),
                (tspec, fspec, S_dB),
                (t_tr, tr_rms, tr_crest),
            ) = res

            self.current_signal = (yw, fs)
            self.current_metrics = metrics
            self.spec_cache = (tspec, fspec, S_dB)
            self.trend_cache = (t_tr, tr_rms, tr_crest)

            # Grafikler
            self.tab_wave.plot_wave(t_sig, yw)
            self.tab_psd.plot_psd(f, Pxx, metrics.tpf_hz)
            self.tab_env.plot_env(fe, Pe)
            self.tab_spec.plot_spec(tspec, fspec, S_dB)
            self.tab_trend.plot_trend(t_tr, tr_rms, tr_crest)

            # Özet (kural tabanlı)
            self.update_summary(metrics, is_ae=(p.source == "ae"))

            # ML ovma skoru (model yüklüyse)
            ovma_score = self.predict_ovma_score(metrics, p)
            if ovma_score is not None:
                self.show_ovma_score_warning(ovma_score)

            QtWidgets.QMessageBox.information(self, "Analiz", "Analiz tamamlandı.")
            print("=== Analiz tamamlandı ===")

        except Exception as e:
            tb = traceback.format_exc()
            print("=== run_analysis HATA ===")
            print(tb)
            QtWidgets.QMessageBox.critical(
                self,
                "Hata",
                f"Hata türü: {type(e).__name__}\nMesaj: {e}",
            )

    def update_summary(self, m: Metrics, is_ae: bool):
        def fmt_pow(v):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "-"
            if v == 0:
                return "0"
            return f"{v:.2e}"

        bp_txt = ", ".join([f"{k}={fmt_pow(v)}" for k, v in m.band_powers.items()])
        r_txt = ", ".join(
            [f"{k}:{v:.1f} dB" for k, v in (m.R_h_dB or {}).items() if np.isfinite(v)]
        ) or "-"
        tpf_txt = f"{m.tpf_hz:.1f} Hz" if m.tpf_hz else "-"

        lines = [
            f"fs = {m.fs_Hz:,.0f} Hz",
            f"RMS = {m.rms_V:.3e} V, Peak = {m.peak_V:.3e} V, Crest = {m.crest:.2f}, Kurtosis = {m.kurtosis:.2f}",
            f"PSD tepe = {m.f_peak_hz:,.0f} Hz, -3 dB bant = {m.bw_3db_hz:,.0f} Hz",
            f"Band güçleri: {bp_txt}",
            f"Zarf tepe = {m.env_mod_peak_hz:,.0f} Hz, env_rms = {m.env_rms:.3e}",
            f"TPF = {tpf_txt}, Yan bant oranları = {r_txt}",
        ]

        interpret = []
        if m.crest > 1.8 or m.kurtosis >= 8:
            interpret.append("Darbeli olay / çentik etkisi olası (crest veya kurtosis yüksek).")
        if is_ae:
            if (m.band_powers.get("band2", 0) + m.band_powers.get("band3", 0)) > (
                m.band_powers.get("band1", 0) * 1.5
            ):
                interpret.append("Yüksek bant güçlerinde artış: ovma/aşınma eğilimi olabilir.")
        else:
            if m.band_powers.get("band3", 0) > (m.band_powers.get("band2", 0) * 1.5):
                interpret.append("8 kHz üstü bant güçleri yüksek: sert yüksek frekans bileşenleri.")
        if m.R_h_dB:
            vals = [v for v in m.R_h_dB.values() if np.isfinite(v)]
            if vals and sum(v >= 6.0 for v in vals[:3]) >= 2:
                interpret.append("TPF çevresinde belirgin yan bantlar: tırlama uyarısı.")
        if not interpret:
            interpret.append("Belirgin risk sinyali yok (kurallar temel seviyede).")

        txt = "\n".join(lines + ["", "Yorum:"] + ["- " + s for s in interpret])
        self.txt_summary.setPlainText(txt)

        color = {"OK": "#c7f0c2", "Dikkat": "#fff3b0", "Uyarı": "#ffc9c9"}.get(
            m.risk_label, "#ddd"
        )
        self.lbl_risk.setText(f"Risk: {m.risk_label}  ({m.risk_score:.0f}/100)")
        self.lbl_risk.setStyleSheet(
            f"QLabel {{font-weight:600; padding:6px; border-radius:8px; background:{color}}}"
        )

    # ----------------- Referans -----------------
    def set_reference_from_current(self):
        if not self.current_metrics or not self.current_signal:
            QtWidgets.QMessageBox.information(
                self, "Referans", "Önce bir analiz çalıştırınız."
            )
            return
        self.ref_metrics = self.current_metrics
        y, fs = self.current_signal
        t = np.arange(len(y)) / fs
        self.ref_wave = (t, y)
        if self.last_psd:
            self.ref_psd = self.last_psd
        if self.last_env:
            self.ref_env = self.last_env
        if self.trend_cache:
            t_tr, tr_rms, tr_crest = self.trend_cache
            self.ref_trend = (t_tr, tr_rms, tr_crest)

        QtWidgets.QMessageBox.information(
            self, "Referans", "Mevcut kayıt referans olarak ayarlandı."
        )

    def live_update_warning(self, m: Metrics):
        if not self.ref_metrics:
            return

        def ratio(a, b):
            if b is None or b == 0:
                return np.nan
            return a / b

        hf_now = m.band_powers.get("band3", 0.0)
        hf_ref = self.ref_metrics.band_powers.get("band3", 0.0)
        hf_ratio = ratio(hf_now, hf_ref)

        rms_ratio = ratio(m.rms_V, self.ref_metrics.rms_V)
        crest_ratio = ratio(m.crest, self.ref_metrics.crest)

        messages = []
        if np.isfinite(hf_ratio) and hf_ratio > 2.0:
            messages.append(
                "HF bandı referansa göre 2 katından fazla: sert darbe/aşınma artışı."
            )
        if np.isfinite(rms_ratio) and rms_ratio > 1.5:
            messages.append(
                "RMS seviyesi referansa göre %50'den fazla yüksek: kesme yükü artıyor (ovma riski)."
            )
        if m.R_h_dB:
            vals = [v for v in m.R_h_dB.values() if np.isfinite(v)]
            if vals and max(vals) > 10.0:
                messages.append(
                    "TPF yan bantları kuvvetli: tırlama ihtimali yüksek (canlı)."
                )

        if messages:
            self.lbl_risk.setText("Risk: Uyarı (canlı)")
            self.lbl_risk.setStyleSheet(
                "QLabel {font-weight:600; padding:6px; border-radius:8px; background:#ffc9c9}"
            )
            self.txt_summary.append(
                "\n[CANLI UYARI]\n- " + "\n- ".join(messages)
            )
        else:
            self.lbl_risk.setText("Risk: OK (canlı)")
            self.lbl_risk.setStyleSheet(
                "QLabel {font-weight:600; padding:6px; border-radius:8px; background:#c7f0c2}"
            )

    def play_current_audio(self):
        """Mevcut analiz sinyalini veya seçili ses dosyasını çal / durdur."""
        if not SOUNDDEVICE_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self,
                "Ses çalma",
                "sounddevice paketi kurulu değil (pip install sounddevice).",
            )
            return

        if self.is_playing_audio:
            try:
                sd.stop()
            except Exception:
                pass
            self.is_playing_audio = False
            self.btn_play.setText("Ses Çal")
            return

        y = None
        fs = None

        if self.current_signal is not None:
            y, fs = self.current_signal
        else:
            p = self.gather_params()
            if p.source == "mic" and p.audio_path and Path(p.audio_path).exists():
                y, fs = read_audio_any(p.audio_path)

        if y is None or fs is None:
            QtWidgets.QMessageBox.information(
                self,
                "Ses çalma",
                "Çalınacak sinyal bulunamadı. Önce analiz yapın veya ses dosyası seçin.",
            )
            return

        y = to_mono(np.asarray(y, float))
        max_abs = np.max(np.abs(y)) if len(y) else 0.0
        if max_abs > 0:
            y = 0.9 * y / max_abs

        try:
            sd.stop()
            sd.play(y, int(fs))
            self.is_playing_audio = True
            self.btn_play.setText("Durdur")
        except Exception as e:
            self.is_playing_audio = False
            self.btn_play.setText("Ses Çal")
            QtWidgets.QMessageBox.critical(self, "Ses çalma hatası", str(e))

    # ----------------- Canlı Dinleme -----------------
    def start_live(self):
        if not SOUNDDEVICE_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self,
                "Canlı dinleme",
                "sounddevice paketi kurulu değil (pip install sounddevice).",
            )
            return
        if self.live_stream is not None:
            return

        buf_len = int(self.live_fs * self.live_win_sec * 4)
        self.live_buffer = np.zeros(buf_len, dtype=float)
        self.live_write_idx = 0

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status)
            y = indata[:, 0].astype(float)
            n = len(y)
            idx = self.live_write_idx
            buf = self.live_buffer
            L = len(buf)
            if idx + n <= L:
                buf[idx:idx + n] = y
            else:
                k = L - idx
                buf[idx:] = y[:k]
                buf[: n - k] = y[k:]
            self.live_write_idx = (idx + n) % L

        try:
            self.live_stream = sd.InputStream(
                samplerate=int(self.live_fs),
                channels=1,
                callback=audio_callback,
            )
            self.live_stream.start()
        except Exception as e:
            self.live_stream = None
            QtWidgets.QMessageBox.critical(self, "Canlı dinleme", str(e))
            return

        self.live_timer = QtCore.QTimer(self)
        self.live_timer.timeout.connect(self.update_live_plots)
        self.live_timer.start(150)

        self.lbl_risk.setText("Risk: OK (canlı)")
        self.lbl_risk.setStyleSheet(
            "QLabel {font-weight:600; padding:6px; border-radius:8px; background:#c7f0c2}"
        )
        self.txt_summary.setPlainText(
            "Canlı dinleme başlatıldı.\nReferans tanımlıysa gri eğriler referansı gösterir."
        )

    def stop_live(self):
        if self.live_timer is not None:
            self.live_timer.stop()
            self.live_timer = None
        if self.live_stream is not None:
            try:
                self.live_stream.stop()
                self.live_stream.close()
            except Exception:
                pass
            self.live_stream = None
        if self.live_buffer is not None:
            self.live_buffer = None
        self.txt_summary.append("\nCanlı dinleme durduruldu.")

    def update_live_plots(self):
        if self.live_buffer is None:
            return
        buf = self.live_buffer.copy()
        L = len(buf)
        N = int(self.live_fs * self.live_win_sec)
        if N > L:
            N = L
        idx_end = self.live_write_idx
        idx_start = (idx_end - N) % L
        if idx_start < idx_end:
            y_win = buf[idx_start:idx_end]
        else:
            y_win = np.concatenate([buf[idx_start:], buf[:idx_end]])

        if not np.any(np.abs(y_win) > 1e-8):
            self.lbl_risk.setText("Risk: - (canlı sinyal yok)")
            self.lbl_risk.setStyleSheet(
                "QLabel {font-weight:600; padding:6px; border-radius:8px; background:#ddd}"
            )
            return

        if len(y_win) < 16:
            return

        p = self.gather_params()
        p.source = "mic"
        res = analyze_signal(y_win, self.live_fs, p, is_ae=False)
        metrics, (f, Pxx), (fe, Pe), (t, yw), (tspec, fspec, S_dB), (t_tr, tr_rms, tr_crest) = res

        self.current_signal = (yw, self.live_fs)
        self.current_metrics = metrics
        self.spec_cache = (tspec, fspec, S_dB)
        self.trend_cache = (t_tr, tr_rms, tr_crest)
        self.last_psd = (f, Pxx)
        self.last_env = (fe, Pe)

        wave_ref = self.ref_wave
        psd_ref = self.ref_psd
        env_ref = self.ref_env
        trend_ref = self.ref_trend

        self.tab_wave.plot_wave(t, yw, ref=wave_ref)
        self.tab_psd.plot_psd(f, Pxx, metrics.tpf_hz, ref=psd_ref)
        self.tab_env.plot_env(fe, Pe, ref=env_ref)
        self.tab_spec.plot_spec(tspec, fspec, S_dB)
        self.tab_trend.plot_trend(t_tr, tr_rms, tr_crest, ref=trend_ref)

        self.update_summary(metrics, is_ae=False)
        self.live_update_warning(metrics)

    # ----------------- Save outputs -----------------
    def save_outputs(self):
        if not self.current_metrics or not self.current_signal:
            QtWidgets.QMessageBox.information(
                self, "Kaydet", "Önce bir analiz çalıştırınız."
            )
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Çıktı klasörü seç"
        )
        if not out_dir:
            return
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_w: Figure = self.tab_wave.figure
        fig_p: Figure = self.tab_psd.figure
        fig_e: Figure = self.tab_env.figure
        fig_s: Figure = self.tab_spec.figure
        fig_t: Figure = self.tab_trend.figure
        fig_w.savefig(out / "wave.png", dpi=150)
        fig_p.savefig(out / "psd.png", dpi=150)
        fig_e.savefig(out / "envelope.png", dpi=150)
        fig_s.savefig(out / "spectrogram.png", dpi=150)
        fig_t.savefig(out / "trend.png", dpi=150)
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.current_metrics), f, ensure_ascii=False, indent=2)
        with open(out / "report.txt", "w", encoding="utf-8") as f:
            f.write(self.txt_summary.toPlainText())
        QtWidgets.QMessageBox.information(
            self, "Kaydet", f"Çıktılar kaydedildi → {out}"
        )

    # ----------------- Ovma etiketi & ML -----------------
    def save_ovma_label(self):
        """
        Şu an seçili ses dosyası için ovma skorunu labels_reg.csv'ye kaydet/güncelle.
        Dataset oluştururken bu dosyayı kullanacağız.
        """
        from pathlib import Path

        p = self.gather_params()
        if not p.audio_path:
            QtWidgets.QMessageBox.information(
                self, "Ovma skoru", "Önce Mic/Telefon modunda bir ses dosyası seçiniz."
            )
            return

        audio_path = Path(p.audio_path)
        if not audio_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "Ovma skoru", f"Ses dosyası bulunamadı:\n{audio_path}"
            )
            return

        fname = audio_path.name
        rpm = p.rpm or 0.0
        ovma_score = int(self.spin_ovma_label.value())

        labels_path = Path("labels_reg.csv")

        if labels_path.exists():
            df = pd.read_csv(labels_path)
        else:
            df = pd.DataFrame(columns=["file_name", "rpm", "ovma_score"])

        mask = df["file_name"] == fname
        if mask.any():
            df.loc[mask, "rpm"] = rpm
            df.loc[mask, "ovma_score"] = ovma_score
        else:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [{"file_name": fname, "rpm": rpm, "ovma_score": ovma_score}]
                    ),
                ],
                ignore_index=True,
            )

        df.to_csv(labels_path, index=False)
        QtWidgets.QMessageBox.information(
            self,
            "Ovma skoru",
            f"{fname} için ovma skoru kaydedildi: {ovma_score} / 100",
        )

    def load_ovma_model(self):
        """
        Ovma skor regresyon modelini (ovma_reg_model.pkl) ve
        feature kolon listesini (ovma_reg_dataset.csv'den) yükler.
        """
        from pathlib import Path

        model_path = Path("ovma_reg_model.pkl")
        if not model_path.exists():
            print("[ML] ovma_reg_model.pkl bulunamadı, ovma skoru tahmini devre dışı.")
            return

        try:
            self.ovma_model = joblib.load(model_path)
            print("[ML] Ovma modeli yüklendi.")

            ds_path = Path("ovma_reg_dataset.csv")
            if ds_path.exists():
                df = pd.read_csv(ds_path, nrows=1)
                drop_cols = ["ovma_score", "file_name"]
                self.ovma_feature_cols = [
                    c for c in df.columns if c not in drop_cols
                ]
                print("[ML] Feature kolonları:", self.ovma_feature_cols)
            else:
                print("[ML] Uyarı: ovma_reg_dataset.csv bulunamadı, feature kolonları bilinmiyor.")
        except Exception as e:
            print("[ML] Model yüklenemedi:", e)
            self.ovma_model = None
            self.ovma_feature_cols = None

    def predict_ovma_score(self, m: Metrics, p: Params) -> Optional[float]:
        """
        Metrics + Params'tan 0–100 arası ovma skoru tahmin eder.
        Model yoksa veya hata olursa None döner.
        """
        if self.ovma_model is None or not self.ovma_feature_cols:
            return None

        feat = metrics_to_feature_dict(m)

        if p.rpm is not None:
            feat["rpm"] = p.rpm

        x = [feat.get(col, 0.0) for col in self.ovma_feature_cols]

        try:
            y_pred = self.ovma_model.predict([x])[0]
            score = float(max(0.0, min(100.0, y_pred)))
            return score
        except Exception as e:
            print("[ML] Tahmin hatası:", e)
            return None

    def show_ovma_score_warning(self, score: float):
        """
        Summary kutusunda ovma skorunu ve renkli durum etiketini gösterir.
        """
        self.txt_summary.append("")
        self.txt_summary.append(f"[ML] Ovma skoru: {score:.1f} / 100")

        if score < 30:
            durum = "Düşük (normal aralık)"
            color = "#c7f0c2"
        elif score < 60:
            durum = "Orta seviye ovma"
            color = "#fff3b0"
        else:
            durum = "Yüksek ovma – takım/parametre kontrol önerilir"
            color = "#ffc9c9"

        self.txt_summary.append(f"Durum (ML): {durum}")

        self.lbl_risk.setText(f"Ovma Skoru (ML): {score:.0f} / 100 – {durum}")
        self.lbl_risk.setStyleSheet(
            f"QLabel {{font-weight:600; padding:6px; border-radius:8px; background:{color}}}"
        )


# ------------------------- Entry Point ------------------------------ #

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
