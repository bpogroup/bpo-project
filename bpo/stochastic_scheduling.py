import random
from problems import Problem
from planners import Planner
from simulator import Simulator, Reporter
from math import sqrt


class StochasticSchedulingProblem(Problem):
    """
    A specific :class:`.Problem` that represents a simple stochastic scheduling problem, i.e.:
    - it has n cases with 1 task that arrive at time 0.
    - it has 1 resource.
    - task processing time is exponential with some mu that differs per case (we will assign shortest processing time first)
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
        return task.data["P"]

    def interarrival_time_sample(self):
        if not self.all_cases_generated():
            return 0
        else:
            return float("inf")

    def data_sample(self, task_type):
        data = dict()
        data["P"] = random.uniform(1, 5)
        return data

    def all_cases_generated(self):
        return self.nr_cases_generated() >= self._n


class StratifiedStochasticSchedulingProblem(Problem):
    """
    A specific :class:`.Problem` that represents a stratified stochastic scheduling problem,
    i.e. a variant of the stochastic scheduling problem in which:
    - there are two classes of cases, a and b
    - there is a processing time P of the task T
    - there is also a prediction of the processing time, P_hat
    - this prediction has an error that differs per class: error_a, error_b
    - there is also an overall prediction error: error
    - we distinguish between a P_hat that is not stratified, but has the error that is the same for both classes, and a P_hat that is stratified
    The properties for a particular case are stored in the case's task T data, with fields:
    P, P_hat, P_hat_stratified, error, error_a, error_b, class
    """

    resources = ["R"]
    task_types = ["T"]

    def __init__(self, n):
        super().__init__()
        self._n = n

    def sample_initial_task_type(self):
        return "T"

    def processing_time_sample(self, resource, task):
        return task.data["P"]

    def interarrival_time_sample(self):
        if not self.all_cases_generated():
            return 0
        else:
            return float("inf")

    def data_sample(self, task_type):
        data = dict()
        data["P"] = random.uniform(1, 5)
        alpha = 0.5  # the fraction of cases of class a
        data["class"] = random.choices(["a", "b"], weights=[alpha, 1-alpha], k=1)[0]
        data["error_a"] = 1  # the error is normally distributed with mean 0 and this variance
        data["error_b"] = 0.5  # the error is normally distributed with mean 0 and this variance
        data["error"] = (alpha**2) * data["error_a"] + ((1-alpha)**2) * data["error_b"]  # the overall (variance of) error can be computed from the errors of the individual classes
        # now we compute P_hat by adding an error sample to P
        class_specific_error_sample = random.normalvariate(0, sqrt(data["error_" + data["class"]]))
        data["P_hat_stratified"] = data["P"] + class_specific_error_sample  # this can lead to negative mu_hat, which is strictly speaking impossible, but since it is just a prediction, it should not matter
        general_error_sample = random.normalvariate(0, sqrt(data["error"]))
        data["P_hat"] = data["P"] + general_error_sample  # this can lead to negative mu_hat, which is strictly speaking impossible, but since it is just a prediction, it should not matter
        return data

    def all_cases_generated(self):
        return self.nr_cases_generated() >= self._n


class SPTPlanner(Planner):
    """
    A :class:`.Planner` that assigns shortest processing time first.

    Note that this planner uses the actual processing time P to determine which task has the shortest processing time.
    Strictly speaking we do not know this actual processing time and we should calculate with P_hat or P_hat_stratified.
    """

    def assign(self, environment):
        if len(environment.available_resources) == 0:
            return []  # for efficiency

        if not environment.problem.all_cases_generated():
            return []  # in stochastic scheduling, we will wait for all cases to arrive before assigning them

        assignments = []
        best_task = None
        best_mu = None
        for task in environment.unassigned_tasks.values():
            if best_task is None or task.data["P"] < best_mu:
                best_task = task
                best_mu = task.data["P"]
        assignments.append((best_task, list(environment.available_resources)[0], environment.now))

        return assignments


if __name__ == "__main__":
    problem = StratifiedStochasticSchedulingProblem(100)
    planner = SPTPlanner()
    reporter = Reporter()
    results = Simulator.replicate(problem, planner, reporter, 1000000, 20)

    print(Reporter.aggregate(results))
