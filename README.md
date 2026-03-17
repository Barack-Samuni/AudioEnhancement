# Audio Quality Evaluation Suite

This project provides a comprehensive set of tools for objective evaluation of speech quality and intelligibility. It allows you to compare a "clean" reference audio file against a "degraded" or "processed" version using industry-standard metrics.

## Features

The suite includes several key objective speech quality metrics:

* **PESQ (Perceptual Evaluation of Speech Quality):** Predicts the subjective quality of speech as heard by a human.
* **STOI (Short-Time Objective Intelligibility):** Measures how understandable the speech is.
* **SNR (Signal-to-Noise Ratio):** A standard physical measure of the signal strength relative to background noise.
* **LLR (Log Likelihood Ratio):** Evaluates the spectral distance between the signals based on LPC coefficients.
* **Composite Metrics (CBAK & COVL):** Based on Loizou's regression formulas to predict background noise quality and overall quality on a MOS scale (1-5).

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

```bash
pip install numpy scipy soundfile pesq pystoi