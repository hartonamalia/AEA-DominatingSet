from ilp_solver import DominatingSetILP
from sa_solver import DominatingSetSA
from visualizations import visualize_graph_with_solution


def run_benchmark(instances, ilp_time_limit=300, sa_time_limit=300, sa_iterations=100):

    results = []

    for instance_data in instances:
        graph, name = instance_data[0], instance_data[1]
        known_solution = instance_data[2] if len(instance_data) > 2 else None

        print(f"\nSolving instance: {name}")
        print(f"Graph size: {len(graph.nodes())} nodes, {len(graph.edges())} edges")

        if known_solution:
            known_solution_size = len(known_solution)
            print(f"Known solution size: {known_solution_size}")

            # Verify the known solution is valid
            ds_checker = DominatingSetSA(graph)
            is_valid_solution = ds_checker.is_dominating_set(known_solution)
            print(f"Known solution is valid: {is_valid_solution}")
        else:
            known_solution_size = None
            is_valid_solution = None

        # Solve using ILP
        print("Solving with ILP...")
        ilp_solver = DominatingSetILP(graph)
        ilp_solution = ilp_solver.solve(time_limit=ilp_time_limit)
        ilp_obj_value = ilp_solver.objective_value
        ilp_runtime = ilp_solver.runtime
        ilp_valid = ilp_solver.objective_value < float('inf')

        # Solve using SA
        print("Solving with SA...")
        sa_solver = DominatingSetSA(graph)
        sa_solution = sa_solver.solve(
            initial_temp=100.0,
            final_temp=0.1,
            cooling_rate=0.95,
            iterations_per_temp=sa_iterations,
            time_limit=sa_time_limit
        )
        sa_obj_value = sa_solver.best_objective_value
        sa_runtime = sa_solver.runtime
        sa_valid = sa_solver.is_dominating_set(sa_solution)

        # Visualize solutions
        if ilp_valid:
            visualize_graph_with_solution(
                graph, ilp_solution,
                f"{name} - ILP Solution (size: {len(ilp_solution)})",
                filename=f"results/{name}_ilp.png"
            )

        if sa_valid:
            visualize_graph_with_solution(
                graph, sa_solution,
                f"{name} - SA Solution (size: {len(sa_solution)})",
                filename=f"results/{name}_sa.png"
            )

        # If we have a known optimal solution, visualize it as well
        if known_solution and is_valid_solution:
            visualize_graph_with_solution(
                graph, known_solution,
                f"{name} - Known Solution (size: {known_solution_size})",
                filename=f"results/{name}_known.png"
            )

        # Record results
        result_dict = {
            'instance': name,
            'nodes': len(graph.nodes()),
            'edges': len(graph.edges()),
            'ilp_solution_size': len(ilp_solution) if ilp_valid else None,
            'ilp_runtime': ilp_runtime,
            'ilp_valid': ilp_valid,
            'sa_solution_size': len(sa_solution) if sa_valid else None,
            'sa_runtime': sa_runtime,
            'sa_valid': sa_valid,
        }

        # Add known solution info if available
        if known_solution_size is not None:
            result_dict['known_solution_size'] = known_solution_size
            result_dict['known_solution_valid'] = is_valid_solution

            if ilp_valid:
                result_dict['ilp_gap'] = ((
                                                  abs(len(ilp_solution) - known_solution_size) )/ known_solution_size * 100) if known_solution_size > 0 else None

            if sa_valid:
                result_dict['sa_gap'] = ((
                                                 abs(len(sa_solution) - known_solution_size) )/ known_solution_size * 100) if known_solution_size > 0 else None

        # Add gap between ILP and SA if both valid
        if ilp_valid and sa_valid:
            result_dict['ilp_sa_gap'] = ((len(sa_solution) - len(ilp_solution)) / len(ilp_solution) * 100) if len(
                ilp_solution) > 0 else None

        results.append(result_dict)

        print(
            f"ILP: {'Valid' if ilp_valid else 'Invalid'}, Size: {len(ilp_solution) if ilp_valid else 'N/A'}, Time: {ilp_runtime:.2f}s")
        print(
            f"SA: {'Valid' if sa_valid else 'Invalid'}, Size: {len(sa_solution) if sa_valid else 'N/A'}, Time: {sa_runtime:.2f}s")
        if known_solution_size is not None:
            print(f"Known: {'Valid' if is_valid_solution else 'Invalid'}, Size: {known_solution_size}")

    return results
