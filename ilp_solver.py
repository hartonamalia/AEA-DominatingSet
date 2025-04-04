import pulp as pl
import time


class DominatingSetILP:
    def __init__(self, graph):

        self.graph = graph
        self.solution = None
        self.objective_value = None
        self.runtime = None

    def solve(self, time_limit=300):

        start_time = time.time()
        self._solve_with_pulp(time_limit)
        self.runtime = time.time() - start_time
        return self.solution

    def _solve_with_pulp(self, time_limit):
        # Create the model
        model = pl.LpProblem(name="dominating_set", sense=pl.LpMinimize)

        # Create variables
        x = {node: pl.LpVariable(name=f"x_{node}", cat='Binary') for node in self.graph.nodes()}

        # Set objective: minimize the number of nodes in the dominating set
        model += pl.lpSum(x[node] for node in self.graph.nodes())

        # Add constraints: each node must be dominated
        for node in self.graph.nodes():
            # A node is dominated if either it or one of its neighbors is in the set
            neighbors = list(self.graph.neighbors(node))
            model += (x[node] + pl.lpSum(x[neigh] for neigh in neighbors) >= 1, f"dominate_{node}")

        # Set time limit
        solver = pl.PULP_CBC_CMD(timeLimit=time_limit)

        # Solve the model
        model.solve(solver)

        if model.status == pl.LpStatusOptimal or model.status == pl.LpStatusNotSolved:
            # Retrieve solution
            self.solution = [node for node in self.graph.nodes() if pl.value(x[node]) > 0.5]
            self.objective_value = len(self.solution)
        else:
            self.solution = []
            self.objective_value = float('inf')

