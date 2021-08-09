from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta


class EventType(Enum):
    CASE_ARRIVAL = 0
    START_TASK = 1
    COMPLETE_TASK = 2
    PLAN_TASKS = 3
    TASK_ACTIVATE = 4
    TASK_PLANNED = 5
    COMPLETE_CASE = 6


class TimeUnit(Enum):
    SECONDS = 0
    MINUTES = 1
    HOURS = 2
    DAYS = 3


class Event:
    def __init__(self, event_type, moment, task, resource=None, nr_tasks=0, nr_resources=0):
        self.event_type = event_type
        self.moment = moment
        self.task = task
        self.resource = resource
        self.nr_tasks = nr_tasks
        self.nr_resources = nr_resources

    def __lt__(self, other):
        return self.moment < other.moment

    def __str__(self):
        return str(self.event_type) + "\t(" + str(round(self.moment, 2)) + ")\t" + str(self.task) + "," + str(self.resource)


class ReporterElement(ABC):
    @abstractmethod
    def restart(self):
        raise NotImplementedError

    @abstractmethod
    def report(self, event):
        raise NotImplementedError

    @abstractmethod
    def summarize(self):
        raise NotImplementedError


class TasksReporterElement(ReporterElement):
    def __init__(self):
        self.nr_tasks_completed = 0
        self.nr_tasks_started = 0
        self.processing_time = 0
        self.waiting_time = 0
        self.task_start_times = dict()
        self.task_activation_times = dict()

    def restart(self):
        self.__init__()

    def report(self, event):
        if event.event_type == EventType.TASK_ACTIVATE:
            self.task_activation_times[event.task.id] = event.moment
        elif event.event_type == EventType.START_TASK:
            self.task_start_times[event.task.id] = event.moment
            if event.task.id in self.task_activation_times.keys():
                self.nr_tasks_started += 1
                self.waiting_time += event.moment - self.task_activation_times[event.task.id]
                del self.task_activation_times[event.task.id]
        elif event.event_type == EventType.COMPLETE_TASK:
            if event.task.id in self.task_start_times.keys():
                self.nr_tasks_completed += 1
                self.processing_time += event.moment - self.task_start_times[event.task.id]
                del self.task_start_times[event.task.id]

    def summarize(self):
        return [("tasks completed", self.nr_tasks_completed), ("task proc time", self.processing_time/self.nr_tasks_completed), ("task wait time", self.waiting_time/self.nr_tasks_started)]


class CaseReporterElement(ReporterElement):
    def __init__(self):
        self.nr_cases_completed = 0
        self.cycle_time = 0
        self.case_start_times = dict()

    def restart(self):
        self.__init__()

    def report(self, event):
        if event.event_type == EventType.CASE_ARRIVAL:
            self.case_start_times[event.task.case_id] = event.moment
        if event.event_type == EventType.COMPLETE_CASE and event.task.case_id in self.case_start_times.keys():
            self.nr_cases_completed += 1
            self.cycle_time += event.moment - self.case_start_times[event.task.case_id]
            del self.case_start_times[event.task.case_id]

    def summarize(self):
        return [("cases completed", self.nr_cases_completed), ("case cycle time", self.cycle_time/self.nr_cases_completed)]


class EventLogReporterElement(ReporterElement):
    def __init__(self, filename, timeunit=TimeUnit.SECONDS, initial_time=datetime(2020, 1, 1), time_format="%Y-%m-%d %H:%M:%S.%f"):
        self.task_start_times = dict()
        self.timeunit = timeunit
        self.initial_time = initial_time
        self.time_format = time_format
        self.logfile = open(filename, "wt")
        self.logfile.write("case_id,task,resource,start_time,completion_time\n")

    def restart(self):
        raise NotImplementedError

    def report(self, event):
        def displace(time):
            return self.initial_time + (timedelta(seconds=time) if self.timeunit == TimeUnit.SECONDS else timedelta(minutes=time) if self.timeunit == TimeUnit.MINUTES else timedelta(hours=time) if self.timeunit == TimeUnit.HOURS else timedelta(days=time) if self.timeunit == TimeUnit.DAYS else None)
        if event.event_type == EventType.START_TASK:
            self.task_start_times[event.task.id] = event.moment
        elif event.event_type == EventType.COMPLETE_TASK and event.task.id in self.task_start_times.keys():
            self.logfile.write(str(event.task.case_id) + ",")
            self.logfile.write(str(event.task.task_type) + ",")
            self.logfile.write(str(event.resource) + ",")
            self.logfile.write(displace(self.task_start_times[event.task.id]).strftime(self.time_format) + ",")
            self.logfile.write(displace(event.moment).strftime(self.time_format) + "\n")
            self.logfile.flush()
            del self.task_start_times[event.task.id]

    def summarize(self):
        self.logfile.close()


class Reporter:
    def __init__(self, warmup=0, reporters=None):
        self.warmup = warmup
        if reporters is None:
            default_reporters = [TasksReporterElement(), CaseReporterElement()]
            self.reporters = default_reporters
        else:
            self.reporters = reporters

    def restart(self):
        for reporter in self.reporters:
            reporter.restart()

    def report(self, event):
        if event.moment > self.warmup:
            for reporter in self.reporters:
                reporter.report(event)

    def summarize(self):
        result = dict()
        for reporter in self.reporters:
            summary = reporter.summarize()
            for summary_line in summary:
                result[summary_line[0]] = summary_line[1]
        return result

    @staticmethod
    def aggregate(summaries):
        nr_replications = len(summaries)
        result = summaries[0].copy()
        for i in range(1, nr_replications):
            for k in summaries[i].keys():
                result[k] += summaries[i][k]
        for k in result.keys():
            result[k] /= nr_replications
        return result


class Simulator:

    def __init__(self, problem, reporter, planner):
        self.events = []

        self.unassigned_tasks = dict()  # a dict task.id -> task
        self.assigned_tasks = dict()  # an assignment is a dict task.id -> (task, resource, start), where start is the moment at which the resource starts processing the task
        self.available_resources = set()
        self.busy_resources = dict()  # resource -> (task, start)
        self.busy_cases = dict()  # case_id -> [active task_id]
        self.reserved_resources = dict()  # resource -> (task, start)
        self.now = 0

        self.reporter = reporter
        self.planner = planner
        self.problem = problem

        self.init_simulation()

    def init_simulation(self):
        # set all resources to available
        for r in self.problem.resources:
            self.available_resources.add(r)

        # reset the problem
        self.problem.restart()

        # generate arrival event for the first task of the first case
        (t, task) = self.problem.next_case()

        self.events.append((t, Event(EventType.CASE_ARRIVAL, t, task)))

    def simulate(self, running_time):
        # repeat until the end of the simulation time:
        while self.now <= running_time:
            # get the first event e from the events
            event = self.events.pop(0)
            # t = time of e
            self.now = event[0]
            event = event[1]
            self.reporter.report(event)

            # if e is an arrival event:
            if event.event_type == EventType.CASE_ARRIVAL:
                # add new task
                self.unassigned_tasks[event.task.id] = event.task
                self.reporter.report(Event(EventType.TASK_ACTIVATE, self.now, event.task))
                self.busy_cases[event.task.case_id] = [event.task.id]
                # generate a new planning event to start planning now for the new task
                self.events.append((self.now, Event(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                # generate a new arrival event for the first task of the next case
                (t, task) = self.problem.next_case()
                self.events.append((t, Event(EventType.CASE_ARRIVAL, t, task)))
                self.events.sort()

            # if e is a start event:
            elif event.event_type == EventType.START_TASK:
                # create a complete event for task
                t = self.now + self.problem.processing_time(event.task, event.resource)
                self.events.append((t, Event(EventType.COMPLETE_TASK, t, event.task, event.resource)))
                self.events.sort()
                # set resource to busy
                del self.reserved_resources[event.resource]
                self.busy_resources[event.resource] = (event.task, self.now)

            # if e is a complete event:
            elif event.event_type == EventType.COMPLETE_TASK:
                # set resource to available
                del self.busy_resources[event.resource]
                self.available_resources.add(event.resource)
                # remove task from assigned tasks
                del self.assigned_tasks[event.task.id]
                self.busy_cases[event.task.case_id].remove(event.task.id)
                # generate unassigned tasks for each next task
                for next_task in event.task.next_tasks:
                    self.unassigned_tasks[next_task.id] = next_task
                    self.reporter.report(Event(EventType.TASK_ACTIVATE, self.now, next_task))
                    self.busy_cases[event.task.case_id].append(next_task.id)
                if len(self.busy_cases[event.task.case_id]) == 0:
                    self.events.append((self.now, Event(EventType.COMPLETE_CASE, self.now, event.task)))
                # generate a new planning event to start planning now for the newly available resource and next tasks
                self.events.append((self.now, Event(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                self.events.sort()

            # if e is a planning event: do assignment
            elif event.event_type == EventType.PLAN_TASKS:
                # there only is an assignment if there are free resources and tasks
                if len(self.unassigned_tasks) > 0 and len(self.available_resources) > 0:
                    assignments = self.planner.assign(self)
                    # for each newly assigned task:
                    for (task, resource, moment) in assignments:
                        # create start event for task
                        self.events.append((moment, Event(EventType.START_TASK, moment, task, resource)))
                        self.reporter.report(Event(EventType.TASK_PLANNED, self.now, task))
                        # assign task
                        del self.unassigned_tasks[task.id]
                        self.assigned_tasks[task.id] = (task, resource, moment)
                        # reserve resource
                        self.available_resources.remove(resource)
                        self.reserved_resources[resource] = (event.task, moment)
                    self.events.sort()

    @staticmethod
    def replicate(problem_instances, planner, reporter, simulation_time):
        summaries = []
        for problem_instance in problem_instances:
            reporter.restart()
            simulator = Simulator(problem_instance, reporter, planner)
            simulator.simulate(simulation_time)
            summaries.append(reporter.summarize())
        return Reporter.aggregate(summaries)
