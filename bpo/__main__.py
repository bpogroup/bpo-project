from problems import ImbalancedProblem, SequentialProblem, MMcProblem, MinedProblem
from simulator import Simulator, Reporter, EventLogReporterElement, TimeUnit
from planners import GreedyPlanner, HeuristicPlanner, ImbalancedPredictivePlanner, PredictiveHeuristicPlanner
from predicters import ImbalancedPredicter, MeanPredicter
from visualizers import boxplot, line_with_ci
from distributions import DistributionType
from miners import mine_problem
import pandas
import numpy as np


def try_mmc():
    planner = GreedyPlanner()
    reporter = Reporter(10000)
    results = Simulator.replicate(MMcProblem(), planner, reporter, 50000, 20)
    print(Reporter.aggregate(results))


# Simulating several planners and predicters for the imbalanced problem
def try_several_planners():
    print(Reporter.aggregate(Simulator.replicate(ImbalancedProblem(spread=1.0), GreedyPlanner(), Reporter(10000), 50000, 20)))
    print(Reporter.aggregate(Simulator.replicate(ImbalancedProblem(spread=1.0), HeuristicPlanner(), Reporter(10000), 50000, 20)))
    print(Reporter.aggregate(Simulator.replicate(ImbalancedProblem(spread=1.0), ImbalancedPredictivePlanner(ImbalancedPredicter), Reporter(10000), 50000, 20)))


# Comparing two spreads of the imbalanced problem in a box plot
def try_comparison():
    problem_instances = []
    spread10 = Simulator.replicate(ImbalancedProblem(spread=1.0), HeuristicPlanner(), Reporter(10000), 50000, 5)
    problem_instances = []
    spread05 = Simulator.replicate(ImbalancedProblem(spread=0.5), HeuristicPlanner(), Reporter(10000), 50000, 5)
    boxplot({'spread 0.5': spread05['task proc time'], 'spread 1.0': spread10['task proc time']})


# Comparing multiple spreads of the imbalanced problem in a line plot
def try_multiple_comparison():
    results = dict()
    for spread in np.arange(0.5, 1.01, 0.05):
        result = Reporter.aggregate(Simulator.replicate(ImbalancedProblem(spread=spread), HeuristicPlanner(), Reporter(10000), 50000, 5))
        results[spread] = result['task wait time']
    line_with_ci(results)


# Printing a trace for the sequential problem
def try_execution_traces():
    reporter = Reporter(warmup=0, reporter_elements=[EventLogReporterElement("../temp/my_log.csv", TimeUnit.MINUTES)])
    simulator = Simulator(SequentialProblem(), reporter, GreedyPlanner())
    simulator.simulate(1000)


# Mining a problem from an event log and saving it to file
def try_mine_problem():
    log = pandas.read_parquet("../bpo/resources/BPI Challenge 2017 - clean Jan Feb.parquet")
    problem = mine_problem(log, datafields={'ApplicationType': DistributionType.CATEGORICAL, 'LoanGoal': DistributionType.CATEGORICAL, 'RequestedAmount': DistributionType.BETA})
    problem.save("../temp/BPI Challenge 2017 - clean Jan Feb - problem.pickle")


# Load a mined problem from file, simulating it and saving the log
def try_simulate_mined_problem():
    problem = MinedProblem.from_file("../temp/BPI Challenge 2017 - clean Jan Feb - problem.pickle")
    reporter = Reporter(warmup=0, reporter_elements=[EventLogReporterElement("../temp/BPI Challenge 2017 - simulated.csv", TimeUnit.HOURS)])
    simulator = Simulator(problem, reporter, GreedyPlanner())
    simulator.simulate(24)


def main():
    # try_mmc()
    # try_several_planners()
    # try_comparison()
    # try_multiple_comparison()
    # try_execution_traces()
    # try_mine_problem()
    try_simulate_mined_problem()


if __name__ == "__main__":
    main()
