import random
import pickle
from math import factorial
from abc import ABC, abstractmethod


class Task:
    """
    A task.

    :param task_id: the identifier of the task.
    :param case_id: the identifier of the case to which the task belongs.
    :param task_type: the type of the task, i.e. one of the :attr:`.Problem.task_types`.
    :param data: a dictionary with additional data that is the result of the task,
                 each item is a label -> value pair.
    """

    def __init__(self, task_id, case_id, task_type, data):
        self.id = task_id
        self.case_id = case_id
        self.task_type = task_type
        self.data = data
        self.processing_times = dict()  # resource -> processing_time
        self.next_tasks = []

    def add_processing_time(self, resource, processing_time):
        """
        Used when instantiating a task: add the time it will take the specified resource to process this task.

        :meta private:
        """
        self.processing_times[resource] = processing_time

    def add_next_task(self, task):
        """
        Used when instantiating a task: add a task that can be performed in the case (with self.case_id)
        when this task is completed. Note that this means that: for all t in self.next_tasks: self.case_id == t.case_id.

        :meta private:
        """
        self.next_tasks.append(task)

    def __str__(self):
        return self.task_type + "(" + str(self.case_id) + ")_" + str(self.id) + (str(self.data) if len(self.data) > 0 else "")


class Problem(ABC):
    """
    Abstract class that all problems must implement.
    An object of the class is an instance of the problem, which is equivalent to a business process case.
    An object has a next_case_id, which is the next case to arrive for the problem. case_id are sequential,
    starting at 0 for the first case to arrive, 1 for the next case to arrive, etc.
    An object also has a dictionary that maps case_id -> (arrival_time, initial_task), where arrival time
    is the simulation time at which the case will arrive and initial_task is the first :class:`.Task`
    that will be executed for the case.
    """

    @property
    @abstractmethod
    def resources(self):
        """A list of identifiers (typically names) of resources."""
        raise NotImplementedError

    @property
    @abstractmethod
    def task_types(self):
        """A list of identifiers (typically labels) of task types."""
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_task_type(self):
        """An element of :attr:`.Problem.task_types` that is the first to execute in any case."""
        raise NotImplementedError

    @abstractmethod
    def resource_pool(self, task_type):
        """
        Returns for each task_type the subset of resources that can perform tasks of that type.

        :param task_type: one of :attr:`.Problem.task_types`
        :return: a list with elements of :attr:`.Problem.resources`
        """
        raise NotImplementedError

    def __init__(self):
        self.next_case_id = 0
        self.cases = dict()  # case_id -> (arrival_time, initial_task)

    def from_generator(self, duration):
        """
        Instantiates the problem by generating cases, their tasks and the corresponding arrival times, processing times, and data at random.
        Uses the methods interarrival_time_sample, processing_time_sample, data_sample, and next_task_types_sample to
        randomly generate the cases and tasks. These can be implemented in specific problems to obtain the desired behavior.

        :param duration: the latest simulation time at which a new case should arrive.
        :return: an instance of the :class:`.Problem`.
        """
        now = 0
        next_case_id = 0
        next_task_id = 0
        unfinished_tasks = []
        # Instantiate cases at the interarrival time for the duration.
        # Generate the first task for each case, without processing times and next tasks, add them to the unfinished tasks.
        while now < duration:
            at = now + self.interarrival_time_sample()
            task = Task(next_task_id, next_case_id, self.initial_task_type, self.data_sample(self.initial_task_type))
            next_task_id += 1
            unfinished_tasks.append(task)
            self.cases[next_case_id] = (at, task)
            next_case_id += 1
            now = at
        # Finish the tasks by:
        # 1. generating the processing times.
        # 2. generating the next tasks, without processing times and next tasks, add them to the unfinished tasks.
        while len(unfinished_tasks) > 0:
            task = unfinished_tasks.pop(0)
            for r in self.resources:
                pt = self.processing_time_sample(r, task)
                task.add_processing_time(r, pt)
            for tt in self.next_task_types_sample(task):
                new_task = Task(next_task_id, task.case_id, tt, self.data_sample(tt))
                next_task_id += 1
                unfinished_tasks.append(new_task)
                task.add_next_task(new_task)
        return self

    @classmethod
    def from_file(cls, filename):
        """
        Instantiates the problem by reading it from file.

        :param filename: the name of the file from which to read the problem.
        :return: an instance of the :class:`.Problem`.
        """
        with open(filename, 'rb') as handle:
            instance = pickle.load(handle)
        return instance

    def save_instance(self, filename):
        """
        Saves the problem to file.

        :param filename: the name of the file to save the problem to.
        """
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def processing_time_sample(self, resource, task):
        """
        Randomly samples the duration of the simulation time it will take the resource to perform the task.

        :param resource: one of the :attr:`.Problem.resources` of the problem.
        :param task: a :class:`.Task` that should come from the problem.
        :return: a float representing a duration in simulation time.
        """
        raise NotImplementedError

    @abstractmethod
    def interarrival_time_sample(self):
        """
        Randomly samples the interarrival time of cases.

        :return: a float representing a duration in simulation time.
        """
        raise NotImplementedError

    @abstractmethod
    def data_sample(self, task_type):
        """
        Randomly samples data for the task type.

        :param task_type: one of the :attr:`.Problem.task_types` of the problem.
        :return: a dictionary with additional data that can be stored in :attr:`.Task.data`.
        """
        raise NotImplementedError

    @abstractmethod
    def next_task_types_sample(self, task):
        """
        Randomly samples the task types that will be performed for the case of the specified task, when that task completes.

        :param task: a :class:`.Task` of this problem.
        :return: a sublist of :attr:`.Problem.task_types`.
        """
        raise NotImplementedError

    def restart(self):
        """
        Restarts this problem instance, i.e.: sets the next case to arrive to the first case.
        """
        self.next_case_id = 0

    def next_case(self):
        """
        Returns the next case to arrive.

        :return: (arrival_time, initial_task), where
                arrival_time is the simulation time at which the case arrives, and
                initial_task is the first task to perform for the case.
        """
        try:
            (arrival_time, initial_task) = self.cases[self.next_case_id]
            self.next_case_id += 1
            return arrival_time, initial_task
        except KeyError:
            return None

    @staticmethod
    def processing_time(task, resource):
        """
        Returns the time it will take the resource to perform the task.

        :param resource: one of the :attr:`.Problem.resources` of the problem.
        :param task: a :class:`.Task` that should come from the problem.
        :return: a float representing a duration in simulation time.
        """
        return task.processing_times[resource]


class MMcProblem(Problem):
    """
    A specific :class:`.Problem` that represents an M/M/c queue, i.e.:
    it has one task type, multiple resources and exponential arrival and processing times.
    This problem can be simulated, but it also has a method :meth:`.Problem.waiting_time_analytical`
    to compute the waiting time analytically for comparison.
    """

    initial_task_type = "T"
    resources = ["R" + str(i) for i in range(1, 3)]
    task_types = ["T"]

    def __init__(self):
        super().__init__()
        self.c = len(self.resources)
        self.rate = (1/10) * max(self.c-1, 1)
        self.ep = 9

    def resource_pool(self, task_type):
        return self.resources

    def processing_time_sample(self, resource, task):
        return random.expovariate(1/self.ep)

    def interarrival_time_sample(self):
        return random.expovariate(self.rate)

    def data_sample(self, task_type):
        return dict()

    def next_task_types_sample(self, task):
        return []

    def waiting_time_analytical(self):
        rate = self.rate
        c = self.c
        ep = self.ep
        rho = (rate*ep)/c
        piw = (((c*rho)**c)/factorial(c))/((1-rho) * sum([(((c*rho)**n)/factorial(n)) for n in range(c)]) + (((c*rho)**c)/factorial(c)))
        ew = piw*(1/(1-rho))*(ep/c)
        return ew


class ImbalancedProblem(Problem):
    """
    A specific :class:`.Problem` with two resources that have different processing times for the same task.
    The difference between the performance of the resources is indicated by the 0 <= spread < 2.0, where
    a higher spread means that the performance of the resources is more different. The resource that
    performs better on the task is indicated by the data['optimal_resource'] of that task.
    """
    initial_task_type = "T"
    resources = ["R1", "R2"]
    task_types = ["T"]

    def __init__(self, spread=1.0):
        super().__init__()
        self.spread = spread

    def resource_pool(self, task_type):
        return self.resources

    def processing_time_sample(self, resource, task):
        ep = 18
        if resource == task.data["optimal_resource"]:
            return random.expovariate(1/((1.0-(self.spread/2.0))*ep))
        else:
            return random.expovariate(1/((1.0+(self.spread/2.0))*ep))

    def interarrival_time_sample(self):
        return random.expovariate(1/10)

    def data_sample(self, task_type):
        data = dict()
        data["optimal_resource"] = random.choice(self.resources)
        return data

    def next_task_types_sample(self, task):
        return []


class SequentialProblem(Problem):
    """
    A specific :class:`.Problem` with two resources and two task types. Each case starts
    with a task of type T1. After that is completed a task of type T2 must be processed.
    The resources have different processing times for the tasks. The resource that
    performs better on a task is indicated by the data['optimal_resource'] of that task.
    Resource R1 performs better on task T1 and resource R2 on task T2.
    """
    initial_task_type = "T1"
    resources = ["R1", "R2"]
    task_types = ["T1", "T2"]

    def resource_pool(self, task_type):
        return self.resources

    def processing_time_sample(self, resource, task):
        ep = 18
        if resource == task.data["optimal_resource"]:
            return random.expovariate(1/(0.5*ep))
        else:
            return random.expovariate(1/(1.5*ep))

    def interarrival_time_sample(self):
        return random.expovariate(1/20)

    def data_sample(self, task_type):
        data = dict()
        data["optimal_resource"] = "R" + task_type[1]
        return data

    def next_task_types_sample(self, task):
        if task.task_type == "T1":
            return ["T2"]
        return []
