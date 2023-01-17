from enum import Enum, auto
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from statistics import mean
import scipy.stats as st
import random


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
    SCHEDULE_RESOURCES = auto()
    """Resources are scheduled every full clock tick.

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
    initial_time + simulation_time timeunits. Data can also be reported on by specifying the
    corresponding data fields. The names of these data fields must correspond to names of data fields as
    they appear in the problem.

    :param filename: the name of the file in which the event log must be stored.
    :param timeunit: the :class:`.TimeUnit` of simulation time.
    :param initial_time: a datetime value.
    :param time_format: a datetime formatting string.
    :param data_fields: the data fields to report in the log.
    """
    def __init__(self, filename, timeunit=TimeUnit.SECONDS, initial_time=datetime(2020, 1, 1), time_format="%Y-%m-%d %H:%M:%S.%f", data_fields=[]):
        self.task_start_times = dict()
        self.timeunit = timeunit
        self.initial_time = initial_time
        self.time_format = time_format
        self.data_fields = data_fields
        self.logfile = open(filename, "wt")
        self.logfile.write("case_id,task,resource,start_time,completion_time")
        for df in self.data_fields:
            self.logfile.write(",")
            self.logfile.write(df)
        self.logfile.write("\n")

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
            self.logfile.write(displace(event.moment).strftime(self.time_format))
            for df in self.data_fields:
                self.logfile.write(",")
                self.logfile.write('"' + str(event.task.data[df]) + '"')
            self.logfile.write("\n")
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
        if event.moment >= self.warmup:
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

        self.unassigned_tasks = dict()
        """
        The tasks that are currently not assigned. A dict task.id -> task, where task is an instance of :class:`.Task`.
        A task is in this list if it must still be performed. After a task is completed does not re-appear in this list.
        """
        self.assigned_tasks = dict()
        """
        The tasks that are currently assigned. A dict task.id -> (task, resource, start), where:
         
        * task is an instance of :class:`.Task`. 
        * start is the moment in simulation at which the resource will start or has started processing the task.
        * resource is the label that identifies a resource in the :class:`.Problem`.
        
        """
        self.available_resources = set()
        """
        The set of resources that are currently available. Each resource is a label that identifies a resource in the :class:`.Problem`. 
        """
        self.away_resources = []
        self.away_resources_weights = []
        """
        The set of resources that are currently away (on a break, home, or working in another process) and consequently not available.
        An away resource is a pair (resource, weight), such that the weight is the weight belonging to the resource according to the problem, i.e.:
        away_resources_weights[i] == problem.resource_weights[problem.resources.index(away_resources[i])] 
        """
        self.busy_resources = dict()
        """
        The resources that are currently busy. A dict resource -> (task, start), where:
        
        * task is an instance of :class:`.Task` is the task that the resource is working on.
        * start is the moment in simulation time at which the resource started working on the task.
          
        """
        self.busy_cases = dict()
        """
        The cases of which a task is currently being performed or must still be performed. A dict case_id -> [active task_id]
        that maps case identifiers for which a task exists to the identifiers of those tasks.
        """

        self.reserved_resources = dict()
        """
        The resources that are currently reserved. A dict resource -> (task, start), where:
        
        * resource is the label that identifies a resource in the :class:`.Problem`.
        * task is an instance of :class:`.Task` is the task on which the resource is expected to work. 
        * start is the moment in simulation at which the resource starts processing the task.
        
        """
        self.now = 0
        """
        The current simulation time.
        """

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

        # generate resource scheduling event to start the schedule
        self.events.append((0, Event(EventType.SCHEDULE_RESOURCES, 0, None)))

        # reset the problem
        self.problem.restart()

        # generate arrival event for the first task of the first case
        (t, task) = self.problem.next_case()
        self.events.append((t, Event(EventType.CASE_ARRIVAL, t, task)))

    def desired_nr_resources(self):
        """
        The number of resources that is currently desired to be working now according to the problem schedule.

        :return: a number of resources.
        """
        return self.problem.schedule[int(self.now % len(self.problem.schedule))]

    def working_nr_resources(self):
        """
        The number of resources that is actually on the work floor now, either available, busy or reserved.

        :return: a number of resources.
        """
        return len(self.available_resources) + len(self.busy_resources) + len(self.reserved_resources)

    def simulate(self, running_time):
        """
        Runs the simulation for the instance that was passed in the constructor.
        The simulator generates events (i.e. cases with their associated arrival times, data, and tasks)
        as they are produced by the problem instance. The simulator generates a plan event each time a
        task or resource becomes available. Events are handled by the reporter.

        If the target utilization rate is set to a value (a fraction f) other than None, the simulator will ensure that resources
        are occupied for approximately the targeted fraction f of the time. It does that by making resources busy with 'other
        tasks' (i.e. tasks other than the ones of the problem being simulated, like tasks in another process). These 'other tasks'
        are not specified further and their activity is not logged, but the resources are being kept busy on those tasks.

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
                if self.problem.is_event(event.task.task_type):  # events can start immediately
                    self.events.append((self.now, Event(EventType.START_TASK, self.now, task, None)))
                    del self.unassigned_tasks[task.id]
                    self.assigned_tasks[task.id] = (task, None, self.now)
                else:  # tasks must be planned
                    self.events.append((self.now, Event(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                # generate a new arrival event for the first task of the next case
                next_case = self.problem.next_case()
                if next_case is not None:
                    (t, task) = next_case
                    self.events.append((t, Event(EventType.CASE_ARRIVAL, t, task)))
                    self.events.sort()

            # if e is a start event:
            elif event.event_type == EventType.START_TASK:
                # create a complete event for task
                t = self.now + self.problem.processing_time_sample(event.resource, event.task)
                self.events.append((t, Event(EventType.COMPLETE_TASK, t, event.task, event.resource)))
                self.events.sort()
                if not self.problem.is_event(event.task.task_type):  # for actual tasks (not events)
                    # set resource to busy
                    del self.reserved_resources[event.resource]
                    self.busy_resources[event.resource] = (event.task, self.now)

            # if e is a complete event:
            elif event.event_type == EventType.COMPLETE_TASK:
                if not self.problem.is_event(event.task.task_type):  # for actual tasks (not events)
                    # set resource to available, if it is still desired, otherwise set it to away
                    del self.busy_resources[event.resource]
                    if self.working_nr_resources() <= self.desired_nr_resources():
                        self.available_resources.add(event.resource)
                    else:
                        self.away_resources.append(event.resource)
                        self.away_resources_weights.append(self.problem.resource_weights[self.problem.resources.index(event.resource)])
                # remove task from assigned tasks
                del self.assigned_tasks[event.task.id]
                self.busy_cases[event.task.case_id].remove(event.task.id)
                # notify the tasks as complete to the problem and receive the next tasks
                next_tasks = self.problem.complete_task(event.task)
                # generate unassigned tasks for each next task
                for next_task in next_tasks:
                    self.unassigned_tasks[next_task.id] = next_task
                    self.reporter.report(Event(EventType.TASK_ACTIVATE, self.now, next_task))
                    self.busy_cases[event.task.case_id].append(next_task.id)
                    if self.problem.is_event(next_task.task_type):  # events can start immediately
                        self.events.append((self.now, Event(EventType.START_TASK, self.now, next_task, None)))
                        del self.unassigned_tasks[next_task.id]
                        self.assigned_tasks[next_task.id] = (next_task, None, self.now)
                if len(self.busy_cases[event.task.case_id]) == 0:
                    self.events.append((self.now, Event(EventType.COMPLETE_CASE, self.now, event.task)))
                # generate a new planning event to start planning now for the newly available resource and next tasks
                self.events.append((self.now, Event(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                self.events.sort()

            # if e is a schedule resources event: move resources between available/away,
            # depending to how many resources should be available according to the schedule.
            elif event.event_type == EventType.SCHEDULE_RESOURCES:
                assert self.working_nr_resources() + len(self.away_resources) == len(self.problem.resources)  # the number of resources must be constant
                assert len(self.problem.resources) == len(self.problem.resource_weights)  # each resource must have a resource weight
                assert len(self.away_resources) == len(self.away_resources_weights)  # each away resource must have a resource weight
                if len(self.away_resources) > 0:  # for each away resource, the resource weight must be taken from the problem resource weights
                    i = random.randrange(len(self.away_resources))
                    assert self.away_resources_weights[i] == self.problem.resource_weights[self.problem.resources.index(self.away_resources[i])]
                required_resources = self.desired_nr_resources() - self.working_nr_resources()
                if required_resources > 0:
                    # if there are not enough resources working
                    # randomly select away resources to work, as many as required
                    for i in range(required_resources):
                        random_resource = random.choices(self.away_resources, self.away_resources_weights)[0]
                        # remove them from away and add them to available resources
                        away_resource_i = self.away_resources.index(random_resource)
                        del self.away_resources[away_resource_i]
                        del self.away_resources_weights[away_resource_i]
                        self.available_resources.add(random_resource)
                    # generate a new planning event to put them to work
                    self.events.append((self.now, Event(EventType.PLAN_TASKS, self.now, None, nr_tasks=len(self.unassigned_tasks), nr_resources=len(self.available_resources))))
                    self.events.sort()
                elif required_resources < 0:
                    # if there are too many resources working
                    # remove as many as possible, i.e. min(available_resources, -required_resources)
                    nr_resources_to_remove = min(len(self.available_resources), -required_resources)
                    resources_to_remove = random.sample(self.available_resources, nr_resources_to_remove)
                    for r in resources_to_remove:
                        # remove them from the available resources
                        self.available_resources.remove(r)
                        # add them to the away resources
                        self.away_resources.append(r)
                        self.away_resources_weights.append(self.problem.resource_weights[self.problem.resources.index(r)])
                # plan the next resource schedule event
                self.events.append((self.now+1, Event(EventType.SCHEDULE_RESOURCES, self.now+1, None)))

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
    def replicate(problem, planner, reporter, simulation_time, replications):
        """
        Creates a simulator for each of the problem_instances, using the specified planner and reporter.
        Simulates each problem instance by calling the :meth:`.simulate` method on it using the
        specified simulation time. Returns the list of summaries generated by the reporter, one summary for
        each simulated problem_instance.

        :param problem: an instance of :class:`.Problem` to simulate.
        :param planner: a :class:`.Planner`.
        :param reporter: a :class:`.Reporter`.
        :param simulation_time: the amount of simulation time for which each problem instance should be simulated.
        :param replications: the number of replications to do.
        :return: a list of summaries, generated by the reporter, one for each element of problem_instances.
        """
        summaries = None
        for i in range(replications):
            reporter.restart()
            problem.restart()
            simulator = Simulator(problem, reporter, planner)
            simulator.simulate(simulation_time)
            summary = reporter.summarize()
            if summaries is None:
                summaries = summary.copy()
                for key in summaries:
                    summaries[key] = []
            for key in summary:
                summaries[key].append(summary[key])
        return summaries
