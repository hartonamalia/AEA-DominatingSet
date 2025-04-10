import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_results(results_df, output_dir="results/analysis"):
    os.makedirs(output_dir, exist_ok=True)

    df = results_df.copy()

    valid_results = df[(df['ilp_valid'] == True) & (df['sa_valid'] == True)].copy()

    if valid_results.empty:
        print("No valid results for both ILP and SA solvers. Cannot perform comparative analysis.")
        return

    analyze_solution_quality(valid_results, output_dir)

    analyze_efficiency(valid_results, output_dir)

    generate_comparison_stats(valid_results, output_dir)

    generate_summary_report(valid_results, output_dir)


def analyze_solution_quality(df, output_dir):
    plt.figure(figsize=(12, 8))

    node_groups = df.groupby('nodes').agg({
        'ilp_solution_size': 'mean',
        'sa_solution_size': 'mean'
    }).reset_index()

    node_groups = node_groups.sort_values('nodes')

    plt.plot(node_groups['nodes'], node_groups['ilp_solution_size'], 'b-o', label='ILP')
    plt.plot(node_groups['nodes'], node_groups['sa_solution_size'], 'r-s', label='SA')

    plt.title('Average Solution Size vs. Graph Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Dominating Set Size')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/solution_size_vs_nodes_avg.png")
    plt.close()


def analyze_efficiency(df, output_dir):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(df['nodes'], df['ilp_runtime'], alpha=0.7)

    if len(df) > 2:
        x = df['nodes'].values
        y = df['ilp_runtime'].values
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_range = np.linspace(min(x), max(x), 100)
        plt.plot(x_range, p(x_range), 'r--', label=f'Fit: {z[0]:.2e}x² + {z[1]:.2f}x + {z[2]:.2f}')

    plt.title('ILP Runtime vs. Problem Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.scatter(df['nodes'], df['sa_runtime'], alpha=0.7, color='green')

    if len(df) > 2:
        x = df['nodes'].values
        y = df['sa_runtime'].values
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_range = np.linspace(min(x), max(x), 100)
        plt.plot(x_range, p(x_range), 'r--', label=f'Fit: {z[0]:.2e}x² + {z[1]:.2f}x + {z[2]:.2f}')

    plt.title('SA Runtime vs. Problem Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Runtime (seconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/runtime_vs_problem_size.png")
    plt.close()


def generate_comparison_stats(df, output_dir):

    comparison_data = {
        'ILP Better Than SA (solution size)': (df['ilp_solution_size'] < df['sa_solution_size']).mean() * 100,
        'SA Better Than ILP (solution size)': (df['sa_solution_size'] < df['ilp_solution_size']).mean() * 100,
        'Equal Solutions': (df['sa_solution_size'] == df['ilp_solution_size']).mean() * 100,
        'Avg. ILP Solution Size': df['ilp_solution_size'].mean(),
        'Avg. SA Solution Size': df['sa_solution_size'].mean(),
        'Avg. ILP Runtime': df['ilp_runtime'].mean(),
        'Avg. SA Runtime': df['sa_runtime'].mean(),
        'ILP Faster Than SA': (df['ilp_runtime'] < df['sa_runtime']).mean() * 100,
        'SA Faster Than ILP': (df['sa_runtime'] < df['ilp_runtime']).mean() * 100
    }

    comparison_df = pd.DataFrame.from_dict(comparison_data, orient='index', columns=['Value'])

    comparison_df.to_csv(f"{output_dir}/algorithm_comparison_stats.csv")

    plt.figure(figsize=(10, 8))
    metrics_to_plot = ['ILP Better Than SA (solution size)', 'SA Better Than ILP (solution size)',
                       'Equal Solutions', 'ILP Faster Than SA', 'SA Faster Than ILP']

    if 'ILP Optimal Solutions (%)' in comparison_data:
        metrics_to_plot.extend(['ILP Optimal Solutions (%)', 'SA Optimal Solutions (%)'])

    values = [comparison_data[metric] for metric in metrics_to_plot]

    plt.barh(metrics_to_plot, values, color=sns.color_palette("viridis", len(metrics_to_plot)))
    plt.title('Algorithm Comparison Metrics')
    plt.xlabel('Percentage (%)')
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/algorithm_comparison_metrics.png")
    plt.close()


def generate_summary_report(df, output_dir):

    summary = {
        'Total Instances': len(df),
        'Graph Size Range': f"{df['nodes'].min()} - {df['nodes'].max()} nodes",
        'Avg. ILP Runtime': f"{df['ilp_runtime'].mean():.2f}s",
        'Avg. SA Runtime': f"{df['sa_runtime'].mean():.2f}s",
        'Avg. ILP Solution Size': f"{df['ilp_solution_size'].mean():.2f} nodes",
        'Avg. SA Solution Size': f"{df['sa_solution_size'].mean():.2f} nodes",
    }


    corr_data = {
        'Nodes-ILP Size Correlation': np.corrcoef(df['nodes'], df['ilp_solution_size'])[0, 1],
        'Nodes-SA Size Correlation': np.corrcoef(df['nodes'], df['sa_solution_size'])[0, 1],
        'Nodes-ILP Runtime Correlation': np.corrcoef(df['nodes'], df['ilp_runtime'])[0, 1],
        'Nodes-SA Runtime Correlation': np.corrcoef(df['nodes'], df['sa_runtime'])[0, 1]
    }

    for key, value in corr_data.items():
        summary[key] = f"{value:.4f}"

    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write("======================================\n")
        f.write("Dominating Set Benchmark - Summary Report\n")
        f.write("======================================\n\n")

        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\n======================================\n")
        f.write("Statistical Correlations\n")
        f.write("======================================\n\n")

        for key, value in corr_data.items():
            strength = "strong" if abs(float(value)) > 0.7 else "moderate" if abs(float(value)) > 0.4 else "weak"
            direction = "positive" if float(value) > 0 else "negative"
            f.write(f"{key}: {value} ({direction} {strength} correlation)\n")