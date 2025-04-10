import os

import pandas as pd

from analysis import analyze_results
from benchmark import run_benchmark
from utils import load_ds_verifier_data, set_random_seed, save_results_to_csv
from visualizations import create_extended_visualizations

if __name__ == "__main__":
    set_random_seed(42)

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/analysis", exist_ok=True)

    print("\nLoading DS Verifier data...")
    ds_verifier_data_dir = "ds_verifier_data"
    ds_verifier_instances = load_ds_verifier_data(ds_verifier_data_dir)
    print(f"Total instances for benchmark: {len(ds_verifier_instances)}")

    print("\nRunning benchmarks...")
    results = run_benchmark(
        ds_verifier_instances,
        ilp_time_limit=60,
        sa_time_limit=60,
        sa_iterations=100
    )

    print("\nSaving results...")
    save_results_to_csv(results, "results/dominating_set_results.csv")

    df = pd.DataFrame(results)
    create_extended_visualizations(df)

    print("\nPerforming enhanced analysis...")
    analyze_results(df)

    print("\nAnalysis complete")