from problems import ImbalancedProblem, SequentialProblem, MMcProblem
from simulator import Simulator, Reporter, EventLogReporterElement, TimeUnit
from planners import GreedyPlanner, HeuristicPlanner, PredictivePlanner
from predicters import ImbalancedPredicter, PerfectPredicter
from visualizers import boxplot, line_with_ci
from miners import mine_problem
import pandas
import numpy as np


def try_mmc():
    problem_instances = []
    for i in range(20):
        problem_instances.append(MMcProblem().from_generator(51000))  # Running for longer than the simulation time, so we do not run out of tasks
    planner = GreedyPlanner()
    reporter = Reporter(10000)
    results = Simulator.replicate(problem_instances, planner, reporter, 50000)
    print(Reporter.aggregate(results))


# Simulating several planners and predicters for the imbalanced problem
def try_several_planners():
    problem_instances = []
    for i in range(20):
        problem_instances.append(ImbalancedProblem(spread=1.0).from_generator(51000))  # Running for longer than the simulation time, so we do not run out of tasks
    print(Reporter.aggregate(Simulator.replicate(problem_instances, GreedyPlanner(), Reporter(10000), 50000)))
    print(Reporter.aggregate(Simulator.replicate(problem_instances, HeuristicPlanner(), Reporter(10000), 50000)))
    print(Reporter.aggregate(Simulator.replicate(problem_instances, PredictivePlanner(ImbalancedPredicter), Reporter(10000), 50000)))
    print(Reporter.aggregate(Simulator.replicate(problem_instances, PredictivePlanner(PerfectPredicter), Reporter(10000), 50000)))


# Comparing two spreads of the imbalanced problem in a box plot
def try_comparison():
    problem_instances = []
    for i in range(5):
        problem_instances.append(ImbalancedProblem(spread=1.0).from_generator(51000))  # Running for longer than the simulation time, so we do not run out of tasks
    spread10 = Simulator.replicate(problem_instances, HeuristicPlanner(), Reporter(10000), 50000)
    problem_instances = []
    for i in range(5):
        problem_instances.append(ImbalancedProblem(spread=0.5).from_generator(51000))  # Running for longer than the simulation time, so we do not run out of tasks
    spread05 = Simulator.replicate(problem_instances, HeuristicPlanner(), Reporter(10000), 50000)
    boxplot({'spread 0.5': spread05['task proc time'], 'spread 1.0': spread10['task proc time']})


# Comparing multiple spreads of the imbalanced problem in a line plot
def try_multiple_comparison():
    results = dict()
    for spread in np.arange(0.5, 1.01, 0.05):
        problem_instances = []
        for i in range(5):
            problem_instances.append(ImbalancedProblem(spread=spread).from_generator(51000))  # Running for longer than the simulation time, so we do not run out of tasks
        result = Reporter.aggregate(Simulator.replicate(problem_instances, HeuristicPlanner(), Reporter(10000), 50000))
        results[spread] = result['task wait time']
    line_with_ci(results)


# Printing a trace for the sequential problem
def try_execution_traces():
    problem_instance = SequentialProblem().from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
    reporter = Reporter(warmup=0, reporter_elements=[EventLogReporterElement("../temp/my_log.csv", TimeUnit.MINUTES)])
    simulator = Simulator(problem_instance, reporter, GreedyPlanner())
    simulator.simulate(1000)


# Mining a problem from an event log
def try_mining():
    log = pandas.read_csv("./resources/BPI Challenge 2017 - clean.zip")
    problem = mine_problem(log)


def main():
    # try_mmc()
    # try_several_planners()
    # try_comparison()
    # try_multiple_comparison()
    # try_execution_traces()
    try_mining()


if __name__ == "__main__":
    main()

# TODO: finish the miner: need to process resource/task combinations and set minimum number of times a resource executes a task (must be higher than 1 otherwise no standard deviation); cast the whole thing into a problem that can be simulated
# TODO: simulate the mined BPI 2017 problem, create event log, check if it corresponds to the original event log
# TODO: finalize the rest of the experiments
# TODO: prepare for cluster computer and run many experiments
