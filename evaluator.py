import os
import re
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import (
    si_snr,
    compute_stoi,
    compute_pesq,
    compute_sdr,
    compute_gcc_segment,
)


# -------------------------------------------------
# Filename matching
# -------------------------------------------------
def normalize_name(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"(clean_|enhanced_|noisy_)", "", name)
    name = re.sub(r"_enhanced.*", "", name)
    return name


def match_files(ref_folder, deg_folder):
    ref_files = os.listdir(ref_folder)
    deg_files = os.listdir(deg_folder)

    ref_map = {normalize_name(f): f for f in ref_files}
    deg_map = {normalize_name(f): f for f in deg_files}

    matched = []

    for key in ref_map:
        if key in deg_map:
            matched.append(
                (
                    os.path.join(ref_folder, ref_map[key]),
                    os.path.join(deg_folder, deg_map[key]),
                )
            )

    return matched


# -------------------------------------------------
# Audio loading
# -------------------------------------------------
def load_pair(ref_path, deg_path):
    ref, sr1 = librosa.load(ref_path, sr=None)
    deg, sr2 = librosa.load(deg_path, sr=None)

    if sr1 != sr2:
        deg = librosa.resample(deg, orig_sr=sr2, target_sr=sr1)

    return ref, deg, sr1


# -------------------------------------------------
# Single-pass metrics
# -------------------------------------------------
def compute_all_metrics(ref, deg, sr):
    metrics = {}
    metrics["SI_SNR"] = si_snr(ref, deg)
    metrics["STOI"] = compute_stoi(ref, deg, sr)
    metrics["PESQ"] = compute_pesq(ref, deg, sr)
    metrics["SDR"] = compute_sdr(ref, deg)
    metrics["GCC_delay"] = compute_gcc_segment(ref, deg, sr)
    return metrics


# -------------------------------------------------
# Dataset evaluation
# -------------------------------------------------
def evaluate_dataset(ref_folder, deg_folder, single_pass=True):

    pairs = match_files(ref_folder, deg_folder)
    results = []

    if single_pass:
        # Simple single-pass: load each pair once, compute all metrics
        for ref_path, deg_path in tqdm(pairs, desc="Files (Single pass)"):
            if not os.path.exists(deg_path):
                continue
            try:
                ref, deg, sr = load_pair(ref_path, deg_path)
                metrics = compute_all_metrics(ref, deg, sr)
                metrics["file"] = os.path.basename(ref_path)
                results.append(metrics)
            except Exception as e:
                print("Error:", ref_path, e)

    else:
        # Multi-pass: load all signals first, then compute metrics with progress bars per metric
        refs, degs, srs, filenames = [], [], [], []
        for ref_path, deg_path in tqdm(pairs, desc="Loading audio"):
            try:
                ref, deg, sr = load_pair(ref_path, deg_path)
                refs.append(ref)
                degs.append(deg)
                srs.append(sr)
                filenames.append(os.path.basename(ref_path))
            except Exception as e:
                print("Error loading:", ref_path, e)

        # Prepare metric containers
        metric_names = ["SI_SNR", "STOI", "PESQ", "SDR", "GCC_delay"]
        metrics_dict = {name: [] for name in metric_names}

        # Compute metrics with per-metric progress bars
        for i in tqdm(range(len(refs)), desc="SI_SNR"):
            metrics_dict["SI_SNR"].append(si_snr(refs[i], degs[i]))

        for i in tqdm(range(len(refs)), desc="STOI"):
            metrics_dict["STOI"].append(compute_stoi(refs[i], degs[i], srs[i]))

        for i in tqdm(range(len(refs)), desc="PESQ"):
            metrics_dict["PESQ"].append(compute_pesq(refs[i], degs[i], srs[i]))

        for i in tqdm(range(len(refs)), desc="SDR"):
            metrics_dict["SDR"].append(compute_sdr(refs[i], degs[i]))

        for i in tqdm(range(len(refs)), desc="GCC_delay"):
            metrics_dict["GCC_delay"].append(
                compute_gcc_segment(refs[i], degs[i], srs[i])
            )

        # Build final results
        for i, fname in enumerate(filenames):
            row = {"file": fname}
            for metric in metric_names:
                row[metric] = metrics_dict[metric][i]
            results.append(row)

    return pd.DataFrame(results)