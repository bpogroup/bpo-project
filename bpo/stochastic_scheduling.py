import random
from problems import Problem
from planners import Planner
from simulator import Simulator, Reporter, ResourceReporterElement
from distributions import ErlangDistribution
from visualizers import boxplot


class StochasticSchedulingProblem(Problem):
    """
    A specific :class:`.Problem` that represents a simple stochastic scheduling problem, i.e.:
    - it has n cases with 1 task that arrive at time 0.
    - it has 1 resource.
    When the cases arrive, an event happens first. This event can have a processing time that is dependent on when the task is scheduled.
    I.e.: cases arrive at t=0, if a case is scheduled for t=10, the event takes t=10, after which the task is enabled.
    """

    resources = ["R"]
    task_types = ["E", "T"]

    def __init__(self, n):
        super().__init__()
        self._n = n
        self._scheduled_times = dict()
        self._case_data = dict()

    def is_event(self, task_type):
        return task_type == "E"

    def sample_initial_task_type(self):
        return "E"

    def all_cases_scheduled(self):
        return len(self._scheduled_times) == self._n

    def schedule_case(self, case_id, time):
        self._scheduled_times[case_id] = time

    def nr_jobs(self):
        return self._n

    def processing_time_sample(self, resource, task):
        assert self.all_cases_scheduled()  # this should only happen after all cases are scheduled

        if task.task_type == "E":
            return self._scheduled_times[task.case_id]
        else:
            return task.data["P"]

    def interarrival_time_sample(self):
        if not self.all_cases_generated():
            return 0
        else:
            return float("inf")

    def all_cases_generated(self):
        return self.nr_cases_generated() >= self._n

    def next_task_types_sample(self, task):
        if task.task_type == "E":
            return ["T"]
        else:
            return []

    def data_sample(self, task):
        if task.task_type == "E":
            # We have three types of cases 0, 1, 2 with likelihood weight[i] that a case is of type i.
            # The processing time of a task of case i is Erlang distributed with shape shapes[i] and rate rates[i]
            shapes = [10, 5, 1]
            rates = [1/10, 1/5, 1/1]
            weights = [0.5, 0.4, 0.1]
            data = dict()
            case_type = random.choices([0, 1, 2], weights=weights, k=1)[0]
            distro = ErlangDistribution(shapes[case_type], rates[case_type])
            data["type"] = case_type
            data["mean"] = distro.mean()
            data["variance"] = distro.var()
            data["P"] = distro.sample()
            self._case_data[task.case_id] = data
        return self._case_data[task.case_id]

    def restart(self):
        super().restart()
        self._scheduled_times = dict()
        self._case_data = dict()


class SVTScheduler(Planner):
    """
    A :class:`.Planner` that schedules shortest variance first.

    The constructor takes the sorting criterion. This can be a data element from the problem.
    """

    def __init__(self, sorting_criterion):
        super().__init__()
        self._sorting_criterion = sorting_criterion

    def assign(self, environment):
        # In stochastic scheduling, we will wait for all cases to arrive (i.e. for n events to be in the set of unassigned tasks).
        # Then we schedule.
        if len(environment.unassigned_tasks.values()) == environment.problem.nr_jobs():
            assert len([task for task in environment.unassigned_tasks.values() if environment.problem.is_event(task.task_type)]) == environment.problem.nr_jobs()  # The n unassigned tasks all should all be events.
            # Schedule the cases shortest variance first.
            task_list = [(task.data[self._sorting_criterion], task) for task in environment.unassigned_tasks.values()]
            task_list.sort(key=lambda x: x[0])
            t = 0
            for (_, task) in task_list:
                environment.problem.schedule_case(task.case_id, t)
                t += task.data["mean"]
            # Assign all events (to no resource) and return that assignment.
            return [(task, None, environment.now) for task in environment.unassigned_tasks.values()]

        # Now all cases have arrived and are scheduled. So all events have been assigned.
        # We still need to assign each task when it arrives.
        if environment.problem.all_cases_scheduled():
            assert len([task for task in environment.unassigned_tasks.values() if environment.problem.is_event(task.task_type)]) == 0  # There should be no unassigned events.

            # We still process the tasks shortest variance first.
            assignments = []
            best_task = None
            best_variance = None
            for task in environment.unassigned_tasks.values():
                if best_task is None or task.data[self._sorting_criterion] < best_variance:
                    best_task = task
                    best_variance = task.data[self._sorting_criterion]
            assignments.append((best_task, list(environment.available_resources)[0], environment.now))

            return assignments

        # If not all cases have arrived or not all cases are scheduled:
        # we do not do anything yet.
        return []


if __name__ == "__main__":
    problem = StochasticSchedulingProblem(100)

    stratified_results = Simulator.replicate(problem, SVTScheduler("variance"), Reporter(reporter_elements=[ResourceReporterElement()]), 1000000, 20)
    print(Reporter.aggregate(stratified_results))

    # This is a bit nasty, we are relying here on the mean always being the same, such that the cases are basically sorted at random.
    # It would be better to sort according to the non-stratified variance (of the overall processing time distributions).
    # However, that would also be the same for all cases, so it would lead to the same effect.
    non_stratified_results = Simulator.replicate(problem, SVTScheduler("mean"), Reporter(reporter_elements=[ResourceReporterElement()]), 1000000, 20)
    print(Reporter.aggregate(non_stratified_results))

    # boxplot({'stratified': stratified_results['task T wait time'], 'non-stratified': non_stratified_results['task T wait time']})
