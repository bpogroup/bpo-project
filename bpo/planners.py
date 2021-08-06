from abc import ABC, abstractmethod


class Planner(ABC):

    @abstractmethod
    def assign(self, environment):
        raise NotImplementedError


# Greedy assignment
class GreedyPlanner(Planner):

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


# For each task plans the best available resource, or another resource if the best one is not available
class HeuristicPlanner(Planner):

    def assign(self, environment):
        assignments = []
        available_resources = environment.available_resources.copy()
        unassigned_tasks_to_process = len(environment.unassigned_tasks)
        for task in environment.unassigned_tasks.values():
            if len(available_resources) == 0:
                break  # for efficiency
            if (task.task_type == "T1" and "R1" in available_resources) or (task.task_type == "T2" and "R2" in available_resources):
                # if a perfect match is possible, make it
                resource = "R"+task.task_type[1]
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
            if (task.task_type == "T1" and "R1" in available_resources) or (task.task_type == "T2" and "R2" in available_resources):
                # if a perfect match is possible, make it
                resource = "R"+task.task_type[1]
                available_resources.remove(resource)
                assignments.append((task, resource, environment.now))
            elif (task.task_type == "T1") and ("R1" in environment.busy_resources.keys()) and ("R2" in environment.available_resources) and (self.predicter.predict_remaining_processing_time(environment.problem, "R1", environment.busy_resources["R1"][0], environment.busy_resources["R1"][1], now) + self.predicter.predict_processing_time_task(environment.problem, "R1", task) < self.predicter.predict_processing_time_task(environment.problem, "R2", task)):
                pass
                # if task is T1 and R1 is busy, but R2 is available:
                #   if the predicted remaining processing time of R1 + predicted processing time for R1 < predicted processing time for R2:
                #     do nothing and wait for R1 to become available
            elif (task.task_type == "T2") and ("R2" in environment.busy_resources.keys()) and ("R1" in environment.available_resources) and (self.predicter.predict_remaining_processing_time(environment.problem, "R2", environment.busy_resources["R2"][0], environment.busy_resources["R2"][1], now) + self.predicter.predict_processing_time_task(environment.problem, "R2", task) < self.predicter.predict_processing_time_task(environment.problem, "R1", task)):
                # same for T2 and R2
                pass
            elif unassigned_tasks_to_process <= len(available_resources):
                # if no perfect match is possible anymore, make a match anyway
                resource = available_resources.pop()
                assignments.append((task, resource, environment.now))
            unassigned_tasks_to_process -= 1
        return assignments
