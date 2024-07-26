import random
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


class WrapperPlanner(Planner):
    """
    A planner that simplifies the implementation of a planner by wrapping it as a policy function.
    """
    def __init__(self, policy_function):
        self.policy_function = policy_function

    def assign(self, environment):
        available_resources = list(environment.available_resources)
        tasks = [task for task in environment.unassigned_tasks.values() if not environment.problem.is_event(task.task_type)]
        assignments = self.policy_function(available_resources, tasks, environment)
        assigned_resources = []
        for (task, resource) in assignments:
            if task not in environment.unassigned_tasks.values():
                raise Exception("ERROR: trying to assign a task that is not in the unassigned_tasks.")
            if resource not in environment.available_resources:
                raise Exception("ERROR: trying to assign a resource that is not in available_resources.")
            if resource not in environment.problem.resource_pool(task.task_type):
                raise Exception("ERROR: trying to assign a resource to a task that is not in its resource pool.")
            if resource in assigned_resources:
                raise Exception("ERROR: trying to assign a resource to multiple tasks.")
            assigned_resources.append(resource)
        for event in environment.unassigned_tasks.values():
            if environment.problem.is_event(event.task_type):
                assignments.append((event, None))
        return [(task, resource, environment.now) for task, resource in assignments]


# Greedy assignment
class GreedyPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""

    def assign(self, environment):
        assignments = []
        available_resources = environment.available_resources.copy()
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in environment.unassigned_tasks.values():
            for resource in available_resources:
                if resource in environment.problem.resource_pool(task.task_type):
                    available_resources.remove(resource)
                    assignments.append((task, resource, environment.now))
                    break
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


class PredictiveHeuristicPlanner(Planner):
    """ A :class:`.Planner` that tries to assign a task to the optimal resource.
        This is the resource that is predicted to have the lowest processing time
        on the task. It takes the list of unassigned tasks and the list of available resources
        and creates the assignment with the lowest processing time (in a greedy manner).
        To avoid starvation of tasks, it only looks in the first nr_lookahead tasks and with
        probability epsilon simply assigns the first task. If nr_lookahead is 0, it looks at all tasks."""
    def __init__(self, predicter, nr_lookahead, epsilon):
        self.predicter = predicter
        self.nr_lookahead = nr_lookahead
        self.epsilon = epsilon

    def assign(self, environment):
        nr_lookahead = self.nr_lookahead
        if random.uniform(0, 1) < self.epsilon:
            nr_lookahead = 1
        elif nr_lookahead == 0:
            nr_lookahead = len(environment.unassigned_tasks)

        possible_assignments = []
        for task in list(environment.unassigned_tasks.values())[:nr_lookahead]:
            for resource in environment.available_resources:
                if resource in environment.problem.resource_pool(task.task_type):
                    possible_assignments.append((self.predicter.predict_processing_time_task(environment.problem, resource, task), resource, task))
        possible_assignments.sort(key=lambda pa: (pa[0], pa[1]))

        assignments = []
        while len(possible_assignments) > 0:
            (processing_time, resource, task) = possible_assignments[0]
            assignments.append((task, resource, environment.now))
            possible_assignments = [(p, r, t) for (p, r, t) in possible_assignments if r != resource and t != task]

        return assignments


# Variant of the heuristic planner:
# For each task plans the best available resource, or another resource if the best one is not available
# Defers planning if it is likely that a better resource will be available some time in the future
class ImbalancedPredictivePlanner(Planner):
    """ Specific planner for the :class:`.problem.ImbalancedProblem`, to be used for test purposes.
        A :class:`.Planner` that tries to assign a task to the optimal resource,
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
