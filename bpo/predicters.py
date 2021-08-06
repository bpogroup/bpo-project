from abc import ABC, abstractmethod


class Predicter(ABC):

    @staticmethod
    @abstractmethod
    def predict_processing_time_task(problem, resource, task):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def predict_remaining_processing_time(problem, resource, task, start_time, now):
        raise NotImplementedError


# Predicter for the imbalanced problem
class ImbalancedPredicter(Predicter):

    @staticmethod
    def predict_processing_time_task(problem, resource, task):
        ep = 18
        if resource == task.data["optimal_resource"]:
            return 0.5*ep
        else:
            return 1.5*ep

    @staticmethod
    def predict_remaining_processing_time(problem, resource, task, start_time, now):
        return ImbalancedPredicter.predict_processing_time_task(problem, resource, task)


# Predicter for the imbalanced problem
class PerfectPredicter(Predicter):

    @staticmethod
    def predict_processing_time_task(problem, resource, task):
        return problem.processing_time(task, resource)

    @staticmethod
    def predict_remaining_processing_time(problem, resource, task, start_time, now):
        return start_time + problem.processing_time(task, resource) - now
