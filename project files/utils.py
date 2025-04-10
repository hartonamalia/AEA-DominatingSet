import csv
import glob
import os
import random
import networkx as nx
import numpy as np


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def load_ds_verifier_data(data_dir):

    instances = []

    if os.path.exists(data_dir):

        gr_files = glob.glob(os.path.join(data_dir, "*.gr"))

        for gr_file in gr_files:
            base_name = os.path.basename(gr_file)
            instance_name = os.path.splitext(base_name)[0]
            sol_file = os.path.join(data_dir, f"{instance_name}.sol")

            if os.path.exists(sol_file):
                G = nx.Graph()

                with open(gr_file, 'r') as f:
                    lines = f.readlines()

                    header = lines[0].strip().split()
                    if len(header) >= 3 and header[0] == 'p' and header[1] == 'ds':
                        num_nodes = int(header[2])

                        G.add_nodes_from(range(1, num_nodes + 1))

                        for line in lines[1:]:
                            line = line.strip()
                            if line and not line.startswith('c'):
                                edge = line.split()
                                if len(edge) >= 2:
                                    u, v = int(edge[0]), int(edge[1])
                                    G.add_edge(u, v)

                known_solution = []
                with open(sol_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('c'):
                            try:
                                node = int(line)
                                known_solution.append(node)
                            except ValueError:
                                continue

                instances.append((G, f"ds_verifier_{instance_name}", known_solution))

    else:

        print(f"Warning: Directory '{data_dir}' not found. Creating it...")
        os.makedirs(data_dir, exist_ok=True)

    return instances


def save_results_to_csv(results, filename="dominating_set_results.csv"):
    fieldnames = set()
    for result in results:
        fieldnames.update(result.keys())

    fieldnames = sorted(list(fieldnames))

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {filename}")


