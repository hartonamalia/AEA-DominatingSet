
import time
import random
import math


class DominatingSetSA:
    def __init__(self, graph):
        self.graph = graph
        self.best_solution = None
        self.best_objective_value = float('inf')
        self.runtime = None

    def is_dominating_set(self, solution):
        dominated = set(solution)
        for node in solution:
            dominated.update(self.graph.neighbors(node))
        return len(dominated) == len(self.graph.nodes())

    def objective_function(self, solution):
        if self.is_dominating_set(solution):
            return len(solution)
        else:
            dominated = set(solution)
            for node in solution:
                dominated.update(self.graph.neighbors(node))
            undominated = len(self.graph.nodes()) - len(dominated)

            return len(self.graph.nodes()) + undominated

    def generate_initial_solution(self):
        all_nodes = list(self.graph.nodes())
        solution = []
        while not self.is_dominating_set(solution) and len(solution) < len(all_nodes):
            remaining_nodes = [n for n in all_nodes if n not in solution]
            if not remaining_nodes:
                break
            node = random.choice(remaining_nodes)
            solution.append(node)
        return solution

    def get_neighbor(self, current_solution):
        all_nodes = list(self.graph.nodes())
        non_solution_nodes = [n for n in all_nodes if n not in current_solution]

        if len(current_solution) == 0 or (len(current_solution) < len(all_nodes) and random.random() < 0.4):
            if non_solution_nodes:
                new_solution = current_solution.copy()
                new_solution.append(random.choice(non_solution_nodes))
                return new_solution

        elif len(current_solution) > 1 and random.random() < 0.4:
            new_solution = current_solution.copy()
            new_solution.remove(random.choice(current_solution))
            return new_solution

        elif non_solution_nodes and current_solution:
                new_solution = current_solution.copy()
                node_to_remove = random.choice(current_solution)
                node_to_add = random.choice(non_solution_nodes)
                new_solution.remove(node_to_remove)
                new_solution.append(node_to_add)
                return new_solution

        return current_solution.copy()

    def solve(self, initial_temp=100.0, final_temp=0.1, cooling_rate=0.95, iterations_per_temp=100, time_limit=300):
        start_time = time.time()
        current_solution = self.generate_initial_solution()
        current_objective = self.objective_function(current_solution)
        self.best_solution = current_solution.copy()
        self.best_objective_value = current_objective

        temperature = initial_temp
        iteration = 0

        while temperature > final_temp and time.time() - start_time < time_limit:
            for i in range(iterations_per_temp):
                if time.time() - start_time >= time_limit:
                    break

                neighbor_solution = self.get_neighbor(current_solution)
                neighbor_objective = self.objective_function(neighbor_solution)

                delta = neighbor_objective - current_objective
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_solution = neighbor_solution
                    current_objective = neighbor_objective

                    if current_objective < self.best_objective_value and self.is_dominating_set(current_solution):
                        self.best_solution = current_solution.copy()
                        self.best_objective_value = current_objective

                iteration += 1

            temperature *= cooling_rate

        self.runtime = time.time() - start_time
        return self.best_solution