from abc import ABC, abstractmethod


class Planner(ABC):
    """Abstract class that all planners must implement."""

    @abstractmethod
    def assign(self, environment):
        """
        Assign tasks to resources from the simulation environment.

        :param environment: a :class:`.Simulator`
        :return: [(task, resource, moment)], where
            task is an instance of :class:`.Task`,
            resource is one of :attr:`.Problem.resources`, and
            moment is a number representing the moment in simulation time
            at which the resource must be assigned to the task (typically, this can also be :attr:`.Simulator.now`).
        """
        raise NotImplementedError


# Greedy assignment
class GreedyPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""

    def assign(self, environment):
        assignments = []
        available_resources = environment.available_resources.copy()
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in environment.unassigned_tasks.values():
            if len(available_resources) > 0:
                resource = available_resources.pop()
                assignments.append((task, resource, environment.now))
            else:
                break  # for efficiency purposes
        return assignments


class HeuristicPlanner(Planner):
    """A :class:`.Planner` that takes each task and tries to assign it to the optimal resource.
        The optimal resource must be one of :attr:`.Problem.resources` and specified as
        value of the 'optimal_resource' of :attr:`.Task.data`. If no such resource is
        available, it will assign an arbitrary resource."""

    def assign(self, environment):
        assignments = []
        available_resources = environment.available_resources.copy()
        unassigned_tasks_to_process = len(environment.unassigned_tasks)
        for task in environment.unassigned_tasks.values():
            if len(available_resources) == 0:
                break  # for efficiency
            if task.data["optimal_resource"] in available_resources:
                # if a perfect match is possible, make it
                resource = task.data["optimal_resource"]
                available_resources.remove(resource)
                assignments.append((task, resource, environment.now))
            elif unassigned_tasks_to_process <= len(available_resources):
                # if no perfect match is possible anymore, make a match anyway
                resource = available_resources.pop()
                assignments.append((task, resource, environment.now))
            unassigned_tasks_to_process -= 1
        return assignments


# Variant of the heuristic planner:
# For each task plans the best available resource, or another resource if the best one is not available
# Defers planning if it is likely that a better resource will be available some time in the future
class PredictivePlanner(Planner):
    """A :class:`.Planner` that tries to assign a task to the optimal resource,
        same as the :class:`.HeuristicPlanner`, but failing that will predict
        if it is better to wait with the assignment or assign to a suboptimal resource.
        Specifically, if the optimal resource is not available,
        it will make a prediction, using the passed predicter, to check if the optimal
        resource will be ready in time before it becomes better to assign another resource.
        If so, it will not assign the task."""
    def __init__(self, predicter):
        self.predicter = predicter

    def assign(self, environment):
        assignments = []
        available_resources = environment.available_resources.copy()
        unassigned_tasks_to_process = len(environment.unassigned_tasks)
        now = environment.now
        for task in environment.unassigned_tasks.values():
            if len(available_resources) == 0:
                break  # for efficiency
            if task.data["optimal_resource"] in available_resources:
                # if a perfect match is possible, make it
                resource = task.data["optimal_resource"]
                available_resources.remove(resource)
                assignments.append((task, resource, environment.now))
            elif (task.data["optimal_resource"] == "R1") and ("R1" in environment.busy_resources.keys()) and ("R2" in environment.available_resources) and (self.predicter.predict_remaining_processing_time(environment.problem, "R1", environment.busy_resources["R1"][0], environment.busy_resources["R1"][1], now) + self.predicter.predict_processing_time_task(environment.problem, "R1", task) < self.predicter.predict_processing_time_task(environment.problem, "R2", task)):
                pass
                # if R1 is the optimal resource and R1 is busy, but R2 is available:
                #   if the predicted remaining processing time of R1 + predicted processing time for R1 < predicted processing time for R2:
                #     do nothing and wait for R1 to become available
            elif (task.data["optimal_resource"] == "R2") and ("R2" in environment.busy_resources.keys()) and ("R1" in environment.available_resources) and (self.predicter.predict_remaining_processing_time(environment.problem, "R2", environment.busy_resources["R2"][0], environment.busy_resources["R2"][1], now) + self.predicter.predict_processing_time_task(environment.problem, "R2", task) < self.predicter.predict_processing_time_task(environment.problem, "R1", task)):
                # same for R2
                pass
            elif unassigned_tasks_to_process <= len(available_resources):
                # if no perfect match is possible anymore, make a match anyway
                resource = available_resources.pop()
                assignments.append((task, resource, environment.now))
            unassigned_tasks_to_process -= 1
        return assignments
