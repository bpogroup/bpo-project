from abc import ABC, abstractmethod


class Predicter(ABC):
    """Abstract class that all predicters must implement."""

    @staticmethod
    @abstractmethod
    def predict_processing_time_task(problem, resource, task):
        """
        Predicts the time it will take a resource to perform a task in a specific problem instance.

        :param problem: an instance of a :class:`.Problem`.
        :param resource: one of the :attr:`.Problem.resources` of the problem.
        :param task: a :class:`.Task` that should come from the problem.
        :return: a float representing a duration in simulation time.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def predict_remaining_processing_time(problem, resource, task, start_time, now):
        """
        Predicts the time a resource needs to complete a task that was already started.

        :param problem: an instance of a :class:`.Problem`.
        :param resource: one of the :attr:`.Problem.resources` of the problem.
        :param task: a :class:`.Task` that should come from the problem.
        :param start_time: the simulation time at which the resource started processing the task.
        :param now: the current simulation time.
        :return: a float representing a duration in simulation time.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def predict_next_task(problem, environment):
        """
        Predicts the next task that will arrive.

        :param problem: an instance of a :class:`.Problem`.
        :param environment: an instance of a :class:`.Simulator`.
        :return: a task that represents the most likely task to arrive next.
        """
        raise NotImplementedError


class ImbalancedPredicter(Predicter):
    """A :class:`.Predicter` that predicts for instances of the :class:`.ImbalancedProblem`."""

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

    @staticmethod
    def predict_next_task(problem, environment):
        raise NotImplementedError


class MeanPredicter(Predicter):
    """A :class:`.Predicter` that predicts that the time a resource will take to perform a task
    is the historical mean. Works only for instances of the :class:`.problems.MinedProblem`."""

    @staticmethod
    def predict_processing_time_task(problem, resource, task):
        (mu, sigma) = problem.processing_time_distribution[(task.task_type, resource)]
        return mu

    @staticmethod
    def predict_remaining_processing_time(problem, resource, task, start_time, now):
        raise NotImplementedError

    @staticmethod
    def predict_next_task(problem, environment):
        raise NotImplementedError
