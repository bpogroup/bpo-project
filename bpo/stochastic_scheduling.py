import random
from problems import Problem
from planners import Planner
from simulator import Simulator, Reporter, EventLogReporterElement, TimeUnit
from visualizers import boxplot, line_with_ci
import pandas
import numpy as np


class StochasticSchedulingProblem(Problem):
    """
    A specific :class:`.Problem` that represents a simple stochastic scheduling problem, i.e.:
    - it has n cases with 1 task that arrive at time 0.
    - it has 1 resource.
    - task processing time is exponential with some mu that differs per case (we will assign shortest mu first)
    (task processing times are assigned through data variables)
    """

    resources = ["R"]
    task_types = ["T"]

    def __init__(self, n):
        super().__init__()
        self._n = n

    def sample_initial_task_type(self):
        return "T"

    def processing_time_sample(self, resource, task):
        mu = task.data["mu"]
        return random.expovariate(1/mu)

    def interarrival_time_sample(self):
        if len(self.cases) < self._n:
            return 0
        else:
            return float("inf")

    def data_sample(self, task_type):
        data = dict()
        data["mu"] = random.uniform(1, 5)
        return data

    def all_cases_generated(self):
        return len(self.cases) >= self._n


class SPTPlanner(Planner):
    """A :class:`.Planner` that assigns shortest processing time first."""

    def assign(self, environment):
        if len(environment.available_resources) == 0:
            return []  # for efficiency

        if not environment.problem.all_cases_generated():
            return []  # in stochastic scheduling, we will wait for all cases to arrive before assigning them

        assignments = []
        best_task = None
        best_mu = None
        for task in environment.unassigned_tasks.values():
            if best_task is None or task.data["mu"] < best_mu:
                best_task = task
                best_mu = task.data["mu"]
        assignments.append((best_task, list(environment.available_resources)[0], environment.now))

        return assignments


if __name__ == "__main__":
    problem_instances = []
    for i in range(20):
        problem_instances.append(StochasticSchedulingProblem(100).from_generator(1000000))
    planner = SPTPlanner()
    reporter = Reporter()
    results = Simulator.replicate(problem_instances, planner, reporter, 1000000)
    print(Reporter.aggregate(results))
