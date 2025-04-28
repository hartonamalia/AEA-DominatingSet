import time
import random
import math


class DominatingSetSA:
    def __init__(self, graph):
        self.graph = graph
        self.best_solution = None
        self.best_objective_value = float('inf')
        self.runtime = None
        self.optimal_solution_found = False

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

            return len(self.graph.nodes()) * 10 + undominated

    def generate_initial_solution(self):
        all_nodes = list(self.graph.nodes())
        solution = all_nodes.copy()

        for node in sorted(all_nodes, key=lambda n: self.graph.degree(n)):
            temp_solution = [n for n in solution if n != node]
            if self.is_dominating_set(temp_solution):
                solution = temp_solution

        return solution

    def get_neighbors(self, current_solution, num_neighbors=5):

        all_nodes = list(self.graph.nodes())
        neighbors = []

        for _ in range(num_neighbors):
            non_solution_nodes = [n for n in all_nodes if n not in current_solution]
            strategy = random.random()

            if len(current_solution) == 0 or (len(current_solution) < len(all_nodes) / 2 and strategy < 0.3):
                if non_solution_nodes:
                    new_solution = current_solution.copy()
                    new_solution.append(random.choice(non_solution_nodes))
                    neighbors.append(new_solution)
                    continue

            elif len(current_solution) > 1 and strategy < 0.5:
                redundant_nodes = []
                for node in current_solution:
                    temp_solution = [n for n in current_solution if n != node]
                    if self.is_dominating_set(temp_solution):
                        redundant_nodes.append(node)

                new_solution = current_solution.copy()
                if redundant_nodes:
                    new_solution.remove(random.choice(redundant_nodes))
                elif len(current_solution) > 1:
                    new_solution.remove(random.choice(current_solution))

                neighbors.append(new_solution)
                continue

            elif non_solution_nodes and current_solution:
                new_solution = current_solution.copy()
                node_to_remove = random.choice(current_solution)
                node_to_add = random.choice(non_solution_nodes)
                new_solution.remove(node_to_remove)
                new_solution.append(node_to_add)
                neighbors.append(new_solution)
                continue

            neighbors.append(current_solution.copy())


        unique_neighbors = []
        seen = set()
        for neighbor in neighbors:
            neighbor_tuple = tuple(sorted(neighbor))
            if neighbor_tuple not in seen:
                seen.add(neighbor_tuple)
                unique_neighbors.append(neighbor)

        return unique_neighbors

    def select_best_neighbor(self, neighbors):

        best_neighbor = None
        best_objective = float('inf')

        for neighbor in neighbors:
            objective = self.objective_function(neighbor)
            if objective < best_objective:
                best_neighbor = neighbor
                best_objective = objective

        return best_neighbor, best_objective

    def solve(self, initial_temp=100.0, final_temp=0.1, cooling_rate=0.95, iterations_per_temp=100, time_limit=300,
              num_neighbors=5):
        start_time = time.time()
        current_solution = self.generate_initial_solution()
        current_objective = self.objective_function(current_solution)

        if self.is_dominating_set(current_solution):
            self.best_solution = current_solution.copy()
            self.best_objective_value = current_objective
            self.optimal_solution_found = True
        else:
            self.best_solution = None
            self.best_objective_value = float('inf')
            self.optimal_solution_found = False

        temperature = initial_temp
        iteration = 0
        stagnation_counter = 0
        max_stagnation = 10

        while temperature > final_temp and time.time() - start_time < time_limit and stagnation_counter < max_stagnation:
            improvement_in_this_temp = False

            for i in range(iterations_per_temp):
                if time.time() - start_time >= time_limit:
                    break

                neighbor_solutions = self.get_neighbors(current_solution, num_neighbors)

                for neighbor_solution in neighbor_solutions:
                    neighbor_objective = self.objective_function(neighbor_solution)
                    is_valid_dominating_set = self.is_dominating_set(neighbor_solution)

                    delta = neighbor_objective - current_objective

                    if delta < 0 or random.random() < math.exp(-delta / temperature):
                        current_solution = neighbor_solution
                        current_objective = neighbor_objective

                        if is_valid_dominating_set and (
                                self.best_solution is None or
                                len(neighbor_solution) < len(self.best_solution)
                        ):
                            self.best_solution = neighbor_solution.copy()
                            self.best_objective_value = len(neighbor_solution)
                            self.optimal_solution_found = True
                            improvement_in_this_temp = True

                        break

                iteration += 1

            if not improvement_in_this_temp:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            temperature *= cooling_rate

        if self.best_solution is None or not self.is_dominating_set(self.best_solution):
            self.best_solution = list(self.graph.nodes())
            self.best_objective_value = len(self.best_solution)
            self.optimal_solution_found = False

        self.runtime = time.time() - start_time
        return self.best_solution