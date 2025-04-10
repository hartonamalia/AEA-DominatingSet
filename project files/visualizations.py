import math
import math
import os
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def visualize_graph_with_solution(graph, solution, title, filename=None, optimal_solution_found=True):

    plt.figure(figsize=(16, 14))

    num_nodes = len(graph.nodes())

    if "ds_verifier" in title:
        k_value = 5.0 / math.sqrt(num_nodes)
        pos = nx.spring_layout(graph, seed=42, k=k_value, iterations=150)
    elif num_nodes <= 150:
        k_value = 1.5 / math.sqrt(num_nodes)
        pos = nx.spring_layout(graph, seed=42, k=k_value, iterations=100)
    else:
        pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.8)

    non_solution_nodes = [node for node in graph.nodes() if node not in solution]
    nx.draw_networkx_nodes(graph, pos, nodelist=non_solution_nodes,
                           node_color='lightblue', node_size=max(300, 1200 / math.sqrt(num_nodes)))

    nx.draw_networkx_nodes(graph, pos, nodelist=solution,
                           node_color='red', node_size=max(500, 1800 / math.sqrt(num_nodes)))

    dominated = set()
    for node in solution:
        dominated.update(graph.neighbors(node))
    dominated = [node for node in dominated if node not in solution]
    nx.draw_networkx_nodes(graph, pos, nodelist=dominated,
                           node_color='lightgreen', node_size=max(300, 1200 / math.sqrt(num_nodes)))

    if "ds_verifier" in title or num_nodes <= 70:
        font_size = max(8, 12 - 0.1 * num_nodes)
        nx.draw_networkx_labels(graph, pos, font_size=font_size, font_weight='bold')

    if not optimal_solution_found:
        plt.title(f"{title}\n(No optimal solution found - using all nodes)", fontsize=14, color='red')
    else:
        plt.title(title, fontsize=14)

    plt.axis('off')

    plt.tight_layout(pad=3.0)

    if not optimal_solution_found:
        plt.figtext(0.5, 0.01, "WARNING: No optimal solution found. Using all nodes as fallback.",
                    ha="center", fontsize=12, bbox={"facecolor": "pink", "alpha": 0.5, "pad": 5})

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def create_extended_visualizations(df):


    os.makedirs("results", exist_ok=True)

    if 'known_solution_size' in df.columns:
        plt.figure(figsize=(14, 8))

        valid_known = df[df['known_solution_valid'] == True].copy()
        valid_ilp = valid_known[valid_known['ilp_valid'] == True].copy()
        valid_sa = valid_known[valid_known['sa_valid'] == True].copy()

        if not valid_ilp.empty or not valid_sa.empty:
            all_instances = sorted(list(set(valid_ilp['instance'].tolist() + valid_sa['instance'].tolist())))

            instance_to_idx = {instance: i for i, instance in enumerate(all_instances)}

            x = np.arange(len(all_instances))
            width = 0.25

            known_values = []
            for instance in all_instances:
                instance_df = valid_known[valid_known['instance'] == instance]
                if not instance_df.empty:
                    known_values.append(instance_df['known_solution_size'].values[0])
                else:
                    known_values.append(None)

            ilp_values = []
            for instance in all_instances:
                instance_df = valid_ilp[valid_ilp['instance'] == instance]
                if not instance_df.empty:
                    ilp_values.append(instance_df['ilp_solution_size'].values[0])
                else:
                    ilp_values.append(None)

            sa_values = []
            for instance in all_instances:
                instance_df = valid_sa[valid_sa['instance'] == instance]
                if not instance_df.empty:
                    sa_values.append(instance_df['sa_solution_size'].values[0])
                else:
                    sa_values.append(None)

            known_x = [i for i, v in enumerate(known_values) if v is not None]
            known_y = [v for v in known_values if v is not None]
            ilp_x = [i for i, v in enumerate(ilp_values) if v is not None]
            ilp_y = [v for v in ilp_values if v is not None]
            sa_x = [i for i, v in enumerate(sa_values) if v is not None]
            sa_y = [v for v in sa_values if v is not None]

            plt.bar([x - width for x in known_x], known_y, width, label='Known Solution')
            plt.bar(ilp_x, ilp_y, width, label='ILP Solution')
            plt.bar([x + width for x in sa_x], sa_y, width, label='SA Solution')

            plt.xlabel('Instance')
            plt.ylabel('Dominating Set Size')
            plt.title('Solution Quality Comparison with Known Solutions')
            plt.xticks(x, [name if len(name) < 15 else name[:12] + '...' for name in all_instances], rotation=45,
                       ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig("results/solution_quality_with_known.png")
            plt.close()

        plt.figure(figsize=(12, 6))

        gap_df = df.dropna(subset=['ilp_gap', 'sa_gap'], how='all').copy()

        if not gap_df.empty:
            gap_df = gap_df.sort_values('known_solution_size')

            gap_df['instance_short'] = gap_df['instance'].apply(
                lambda x: x if len(x) < 15 else x[:12] + '...'
            )

            x = np.arange(len(gap_df))
            width = 0.35

            ilp_gaps = gap_df['ilp_gap'].fillna(0).tolist()
            sa_gaps = gap_df['sa_gap'].fillna(0).tolist()

            plt.bar(x - width / 2, ilp_gaps, width, label='ILP Gap %')
            plt.bar(x + width / 2, sa_gaps, width, label='SA Gap %')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

            plt.xlabel('Instance')
            plt.ylabel('Gap to Known Solution (%)')
            plt.title('Solution Quality Gap Compared to Known Solutions')
            plt.xticks(x, gap_df['instance_short'], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig("results/solution_gap_to_known.png")
            plt.close()


    plt.figure(figsize=(10, 6))
    plt.scatter(df['nodes'], df['ilp_solution_size'], label='ILP', alpha=0.7)
    plt.scatter(df['nodes'], df['sa_solution_size'], label='SA', alpha=0.7)

    if 'known_solution_size' in df.columns:
        plt.scatter(df['nodes'], df['known_solution_size'], label='Known', alpha=0.7, marker='x')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Dominating Set Size')
    plt.title('Solution Size vs. Problem Size')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/solution_size_vs_nodes.png")
    plt.close()
