import pulp as pl
import time


class DominatingSetILP:
    def __init__(self, graph):
        self.graph = graph
        self.solution = None
        self.objective_value = None
        self.runtime = None
        self.optimal_solution_found = False

    def solve(self, time_limit=300):
        start_time = time.time()
        self._solve_with_pulp(time_limit)
        self.runtime = time.time() - start_time
        return self.solution

    def _solve_with_pulp(self, time_limit):
        model = pl.LpProblem(name="dominating_set", sense=pl.LpMinimize)

        x = {node: pl.LpVariable(name=f"x_{node}", cat='Binary') for node in self.graph.nodes()}

        model += pl.lpSum(x[node] for node in self.graph.nodes())

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            model += (x[node] + pl.lpSum(x[neigh] for neigh in neighbors) >= 1, f"dominate_{node}")

        solver = pl.PULP_CBC_CMD(timeLimit=time_limit)

        model.solve(solver)

        if model.status == pl.LpStatusOptimal or model.status == pl.LpStatusNotSolved:
            self.solution = [node for node in self.graph.nodes() if pl.value(x[node]) > 0.5]
            self.objective_value = len(self.solution)

            self.optimal_solution_found = True
        else:
            self.solution = list(self.graph.nodes())
            self.objective_value = len(self.solution)
            self.optimal_solution_found = False