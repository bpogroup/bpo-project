import random
import pickle
from math import factorial
from abc import ABC, abstractmethod, abstractproperty


class Task:
    def __init__(self, task_id, task_type, data=None):
        if data is None:
            data = dict()
        self.id = task_id
        self.task_type = task_type
        self.data = data

    def __str__(self):
        return self.task_type + "_" + str(self.id) + (str(self.data) if len(self.data) > 0 else "")


class Problem(ABC):
    @property
    @abstractmethod
    def resources(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def task_types(self):
        raise NotImplementedError

    def __init__(self):
        self.next_task_id = 0
        self.tasks_pickle = None

    # Instantiates the problem by preparing all interarrival and processing times up to the specified duration.
    # When running the same instantiation is run, it produces the same interarrival and processing times.
    @classmethod
    def from_generator(cls, duration):
        instance = cls()
        now = 0
        next_task_id = 0
        instance.tasks_pickle = dict()
        next_arrivals = []
        for tt in instance.task_types:
            at = instance.interarrival_time_sample(tt)
            next_arrivals.append((at, tt))
        next_arrivals.sort()
        while now < duration:
            task_pickle = dict()
            (at, tt) = next_arrivals.pop(0)
            task = Task(next_task_id, tt, dict())
            next_task_id += 1
            task_pickle["task"] = task
            task_pickle["arrival_time"] = at
            for r in instance.resources:
                pt = instance.processing_time_sample(r, task)
                task_pickle[r] = pt
            instance.tasks_pickle[task.id] = task_pickle
            now = at
            iat = instance.interarrival_time_sample(tt)
            next_arrivals.append((now+iat, tt))
            next_arrivals.sort()
        return instance

    @classmethod
    def from_file(cls, filename):
        instance = cls()
        with open(filename, 'rb') as handle:
            instance.tasks_pickle = pickle.load(handle)
        return instance

    def save_instance(self, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(self.tasks_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def processing_time_sample(self, resource, task):
        raise NotImplementedError

    @abstractmethod
    def interarrival_time_sample(self, task_type):
        raise NotImplementedError

    def restart(self):
        self.next_task_id = 0

    def next_task(self):
        try:
            task = self.tasks_pickle[self.next_task_id]["task"]
            self.next_task_id += 1
            return task
        except e:
            return None

    def processing_time(self, task_id, resource):
        return self.tasks_pickle[task_id][resource]

    def arrival_time(self, task_id):
        return self.tasks_pickle[task_id]["arrival_time"]


class MMcProblem(Problem):
    c = 2
    rate = (1/10) * max(c-1, 1)
    ep = 9
    resources = ["R" + str(i) for i in range(1, c+1)]
    task_types = ["T"]

    def processing_time_sample(self, resource, task):
        return random.expovariate(1/MMcProblem.ep)

    def interarrival_time_sample(self, task_type):
        return random.expovariate(MMcProblem.rate)

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
    resources = ["R1", "R2"]
    task_types = ["T1", "T2"]

    def processing_time_sample(self, resource, task):
        ep = 18
        if resource[1] == task.task_type[1]:
            return random.expovariate(1/(0.5*ep))
        else:
            return random.expovariate(1/(1.5*ep))

    def interarrival_time_sample(self, task_type):
        return random.expovariate(1/20)
