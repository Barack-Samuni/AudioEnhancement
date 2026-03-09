from evaluator import evaluate_dataset, load_pair, compute_all_metrics
import pandas as pd


def run_single_example(ref_file, deg_file):

    ref, deg, sr = load_pair(ref_file, deg_file)

    metrics = compute_all_metrics(ref, deg, sr)

    print("\nSingle file evaluation")
    print("----------------------")

    for k, v in metrics.items():
        print(f"{k:10s}: {v:.4f}")


def run_dataset(ref_folder, deg_folder):

    df = evaluate_dataset(
        ref_folder,
        deg_folder,
        single_pass=True
    )

    print("\nMean metrics")
    print("------------")

    print(df.mean(numeric_only=True))

    df.to_csv("evaluation_results.csv", index=False)

    print("\nSaved results to evaluation_results.csv")


if __name__ == "__main__":

    MODE = "single"   # options: "single" or "dataset"

    if MODE == "single":

        ref_file = "clean/example.wav"
        deg_file = "enhanced/example.wav"

        run_single_example(ref_file, deg_file)

    elif MODE == "dataset":

        ref_folder = "clean"
        deg_folder = "enhanced"

        run_dataset(ref_folder, deg_folder)