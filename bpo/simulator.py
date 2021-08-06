from enum import Enum


class EventType(Enum):
    CASE_ARRIVAL = 0
    START_TASK = 1
    COMPLETE_TASK = 2
    PLAN_TASKS = 3
    TASK_ACTIVATE = 4
    TASK_PLANNED = 5


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


class Reporter:

    def __init__(self, warmup=0, verbose=False):
        self.warmup = warmup
        self.verbose = verbose
        self.tasks = dict()
        self.nr_tasks = 0
        self.total_waiting_time = 0
        self.total_processing_time = 0
        self.total_tasks_planned = 0
        self.total_resources_planned = 0
        self.total_plan_events = 0

    def report(self, event):
        if self.verbose:
            print(event)
        if event.event_type == EventType.TASK_ACTIVATE or event.event_type == EventType.START_TASK or event.event_type == EventType.COMPLETE_TASK:
            if event.task.id not in self.tasks.keys():
                self.tasks[event.task.id] = []
            self.tasks[event.task.id].append(event)
            es = self.tasks[event.task.id]
            if len(es) == 3:
                assert es[0].event_type == EventType.TASK_ACTIVATE
                assert es[1].event_type == EventType.START_TASK
                assert es[2].event_type == EventType.COMPLETE_TASK
                if es[0].moment >= self.warmup:
                    self.nr_tasks += 1
                    self.total_waiting_time += es[1].moment - es[0].moment
                    self.total_processing_time += es[2].moment - es[1].moment
        if event.event_type == EventType.PLAN_TASKS:
            self.total_plan_events += 1
            self.total_tasks_planned += event.nr_tasks
            self.total_resources_planned += event.nr_resources

    def summarize(self):
        result = dict()
        result["nr tasks"] = self.nr_tasks
        result["avg waiting time"] = self.total_waiting_time/self.nr_tasks
        result["avg processing time"] = self.total_processing_time/self.nr_tasks
        result["nr plan events"] = self.total_plan_events
        result["avg tasks per plan event"] = self.total_tasks_planned/self.total_plan_events
        result["avg resources per plan event"] = self.total_resources_planned/self.total_plan_events
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
                # generate unassigned tasks for each next task
                for next_task in event.task.next_tasks:
                    self.unassigned_tasks[next_task.id] = next_task
                    self.reporter.report(Event(EventType.TASK_ACTIVATE, self.now, next_task))
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
    def replicate(problem_instances, planner, warmup_time, simulation_time):
        summaries = []
        for problem_instance in problem_instances:
            reporter = Reporter(warmup_time)
            simulator = Simulator(problem_instance, reporter, planner)
            simulator.simulate(simulation_time)
            summaries.append(reporter.summarize())
        return Reporter.aggregate(summaries)
