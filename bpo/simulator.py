from enum import Enum, auto
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from statistics import mean
import scipy.stats as st


class EventType(Enum):
    """An enumeration for the types of event that can happen in the simulator."""
    CASE_ARRIVAL = auto()
    """A case arrives.
    
    :meta hide-value:"""
    START_TASK = auto()
    """A task starts.
    
    :meta hide-value:"""
    COMPLETE_TASK = auto()
    """A task completes.
    
    :meta hide-value:"""
    PLAN_TASKS = auto()
    """An action is performed to assign tasks to resources.
    
    :meta hide-value:"""
    TASK_ACTIVATE = auto()
    """A task becomes ready to perform (but is not assigned to a resource).
    
    :meta hide-value:"""
    TASK_PLANNED = auto()
    """A task is assigned to a resource.
    
    :meta hide-value:"""
    COMPLETE_CASE = auto()
    """A case completes.
    
    :meta hide-value:"""


class TimeUnit(Enum):
    """An enumeration for the unit in which simulation time is measured."""
    SECONDS = auto()
    """
    Measured in seconds.
    
    :meta hide-value:
    """
    MINUTES = auto()
    """
    Measured in minutes.

    :meta hide-value:
    """
    HOURS = auto()
    """
    Measured in hours.

    :meta hide-value:
    """
    DAYS = auto()
    """
    Measured in days.

    :meta hide-value:
    """


class Event:
    """
    A simulation event.

    :param event_type: the :class:`.EventType`.
    :param moment: the moment in simulation time at which the event happens.
    :param task: the task that triggered, or None for event_type == PLAN_TASKS.
    :param resource: the resource that performs the task, or None for event_type not in [START_TASK, COMPLETE_TASK]
    :param nr_tasks: the number of tasks that must be planned, or 0 for event_type != PLAN_TASKS
    :param nr_resources: the number of resources that is available, or 0 for event_type != PLAN_TASKS
    """
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
    """
    Abstract class that must be implemented by each concrete reporter element.
    A reporter element is part of a :class:`.Reporter`. It receives a :meth:`.ReporterElement.report` call,
    each time a simulation event occurs. It can then store information about that event. Once a simulation run
    is completed, it receives a :meth:`.ReporterElement.summarize` call. It must then report aggregate
    information that it stored about the events. Each time a simulation run starts, it receives
    a :meth:`.ReporterElement.restart` call, upon which it must erase all previously stored information.
    """
    @abstractmethod
    def restart(self):
        """
        Is invoked when a simulation run starts. Must erase all information to start a new report.
        """
        raise NotImplementedError

    @abstractmethod
    def report(self, event):
        """
        Is invoked when a simulation event occurs. Can store information about the event.

        :param event: the simulation :class:`.Event` that occurred.
        """
        raise NotImplementedError

    @abstractmethod
    def summarize(self):
        """
        Is invoked when a simulation run ends. Must return aggregate information about stored event information.

        :return: a list of tuples (label, value), where label is a meaningful name for the reported aggregate information,
                 and value is the corresponding information.
        """
        raise NotImplementedError


class TasksReporterElement(ReporterElement):
    """
    A :class:`.ReporterElement` that keeps information about tasks, specifically:
    * tasks completed: the number of tasks that completed during the simulation run.
    * task proc time: the average of the processing times of the completed tasks.
    * task wait time: the average of the waiting times of the completed tasks.
    This information is returned by the :meth:`.TasksReporterElement.summarize` method
    by the specified labels.
    """
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
    """
    A :class:`.ReporterElement` that keeps information about cases, specifically:
    * cases completed: the number of cases that completed during the simulation run.
    * cases cycle time: the average of the cycle times of the completed cases.
    This information is returned by the :meth:`.CaseReporterElement.summarize` method
    by the specified labels.
    """
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
    """
    A :class:`.ReporterElement` that stored the simulation events that occur in an event log.
    The :meth:`.EventLogReporterElement.summarize` method does not return any information.
    As simulation time is a numerical value, some processing is done to convert simulation time
    into a time format that can be read and interpreted by a process mining tool. To that end,
    the timeunit in which simulation time is measured must be passed as well as the
    initial_time moment in calendar time from which the simulation time will be measured.
    A particular simulation_time moment will then be stored in the log as:
    initial_time + simulation_time timeunits

    :param filename: the name of the file in which the event log must be stored.
    :param timeunit: the :class:`.TimeUnit` of simulation time.
    :param initial_time: a datetime value.
    :param time_format: a datetime formatting string.
    """
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
    """
    A Reporter consists of :class:`.ReporterElement` and reports on the information that is kept by
    its elements. Consequently, it does not do much itself, it mainly passes on events to and
    received aggregate information from its elements.

    It receives a :meth:`.Reporter.report` call, each time a simulation event occurs, which it
    forwards to its elements to enable them to store information about that event.
    It receives a :meth:`.ReporterElement.summarize` call when a simulation run completes. It then
    collects the summaries from each of its elements and returns them in one list.
    It receives a :meth:`.ReporterElement.restart` call when a simulation run starts, which it
    forwards to its elements to enable them to restart.

    During the specified warmup time, the reporter will ignore all events.

    :param warmup: a duration in simulation time.
    :param reporter_elements: a list of :class:`.ReporterElement` instances, when None are provided,
                              creates a reporter with a :class:`.TaskReporterElement` and a :class:`.Case ReporterElement`.
    """
    def __init__(self, warmup=0, reporter_elements=None):
        self.warmup = warmup
        if reporter_elements is None:
            default_reporters = [TasksReporterElement(), CaseReporterElement()]
            self.reporters = default_reporters
        else:
            self.reporters = reporter_elements

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
        aggregation = dict()
        for key in summaries:
            avg = mean(summaries[key])
            n = len(summaries[key])
            ci = st.t.interval(0.95, n - 1, loc=avg, scale=st.sem(summaries[key]))
            aggregation[key] = (avg, avg - ci[0])
        return aggregation


class Simulator:
    """
    A Simulator simulates a specified :class:`.Problem` using a specified :class:`.Planner`.
    The results of the simulation are generated using the specified :class:`.Reporter`.
    There are two main entry points into the simulator:
    * :meth:`.simulate`, which simulates the (single) problem instance passed with the constructor; and
    * :meth:`.replicate`, which simulates a collection of problem instances passed via the replicate method itself.
    """
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
        """
        Re-initializes the simulation, such that it can be run again.
        """
        # set all resources to available
        for r in self.problem.resources:
            self.available_resources.add(r)

        # reset the problem
        self.problem.restart()

        # generate arrival event for the first task of the first case
        (t, task) = self.problem.next_case()

        self.events.append((t, Event(EventType.CASE_ARRIVAL, t, task)))

    def simulate(self, running_time):
        """
        Runs the simulation for the instance that was passed in the constructor.
        The simulator generates events (i.e. cases with their associated arrival times, data, and tasks)
        as they are produced by the problem instance. The simulator generates a plan event each time a
        task or resource becomes available. Events are handled by the reporter.

        :param running_time: the amount of simulation time the simulation should be run for.
        """
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
        """
        Creates a simulator for each of the problem_instances, using the specified planner and reporter.
        Simulates each problem instance by calling the :meth:`.simulate` method on it using the
        speficied simulation time. Returns the list of summaries generated by the reporter, one summary for
        each simulated problem_instance.

        :param problem_instances: a list of instances of a :class:`.Problem`.
        :param planner: a :class:`.Planner`.
        :param reporter: a :class:`.Reporter`.
        :param simulation_time: the amount of simulation time for which each problem instance should be simulated.
        :return: a list of summaries, generated by the reporter, one for each element of problem_instances.
        """
        summaries = None
        for problem_instance in problem_instances:
            reporter.restart()
            simulator = Simulator(problem_instance, reporter, planner)
            simulator.simulate(simulation_time)
            summary = reporter.summarize()
            if summaries is None:
                summaries = summary.copy()
                for key in summaries:
                    summaries[key] = []
            for key in summary:
                summaries[key].append(summary[key])
        return summaries
