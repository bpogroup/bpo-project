import random
import pickle
from math import factorial
from abc import ABC, abstractmethod


class Task:
    def __init__(self, task_id, case_id, task_type, data):
        self.id = task_id
        self.case_id = case_id
        self.task_type = task_type
        self.data = data
        self.processing_times = dict()  # resource -> processing_time
        self.next_tasks = []

    def add_processing_time(self, resource, processing_time):
        self.processing_times[resource] = processing_time

    def add_next_task(self, task):
        self.next_tasks.append(task)

    def __str__(self):
        return self.task_type + "(" + str(self.case_id) + ")_" + str(self.id) + (str(self.data) if len(self.data) > 0 else "")


class Problem(ABC):
    @property
    @abstractmethod
    def resources(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def task_types(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_task_type(self):
        raise NotImplementedError

    def __init__(self):
        self.next_case_id = 0
        self.cases = dict()  # case_id -> (arrival_time, initial_task)

    # Instantiates the problem by preparing all interarrival and processing times up to the specified duration.
    # When running the same instantiation is run, it produces the same interarrival and processing times.
    def from_generator(self, duration):
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
        with open(filename, 'rb') as handle:
            instance = pickle.load(handle)
        return instance

    def save_instance(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def processing_time_sample(self, resource, task):
        raise NotImplementedError

    @abstractmethod
    def interarrival_time_sample(self):
        raise NotImplementedError

    @abstractmethod
    def data_sample(self, task_type):
        raise NotImplementedError

    @abstractmethod
    def next_task_types_sample(self, task):
        raise NotImplementedError

    def restart(self):
        self.next_case_id = 0

    def next_case(self):
        try:
            (arrival_time, initial_task) = self.cases[self.next_case_id]
            self.next_case_id += 1
            return arrival_time, initial_task
        except KeyError:
            return None

    @staticmethod
    def processing_time(task, resource):
        return task.processing_times[resource]


class MMcProblem(Problem):
    initial_task_type = "T"
    c = 2
    rate = (1/10) * max(c-1, 1)
    ep = 9
    resources = ["R" + str(i) for i in range(1, c+1)]
    task_types = ["T"]

    def processing_time_sample(self, resource, task):
        return random.expovariate(1/MMcProblem.ep)

    def interarrival_time_sample(self):
        return random.expovariate(MMcProblem.rate)

    def data_sample(self, task_type):
        return dict()

    def next_task_types_sample(self, task):
        return []

    @staticmethod
    def waiting_time_analytical():
        rate = MMcProblem.rate
        c = MMcProblem.c
        ep = MMcProblem.ep
        rho = (rate*ep)/c
        piw = (((c*rho)**c)/factorial(c))/((1-rho) * sum([(((c*rho)**n)/factorial(n)) for n in range(c)]) + (((c*rho)**c)/factorial(c)))
        ew = piw*(1/(1-rho))*(ep/c)
        return ew


class ImbalancedProblem(Problem):
    initial_task_type = "T"
    resources = ["R1", "R2"]
    task_types = ["T"]

    def __init__(self, spread=1.0):
        super().__init__()
        self.spread = spread

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
    initial_task_type = "T1"
    resources = ["R1", "R2"]
    task_types = ["T1", "T2"]

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
