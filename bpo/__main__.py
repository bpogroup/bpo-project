from problems import ImbalancedProblem, SequentialProblem
from simulator import Simulator, Reporter, EventLogReporterElement, TimeUnit
from planners import GreedyPlanner, HeuristicPlanner, PredictivePlanner
from predicters import ImbalancedPredicter, PerfectPredicter
from visualizers import boxplot, line_with_ci
import numpy as np

# Simulating several planners and predicters for the imbalanced problem
problem_instances = []
for i in range(20):
    problem_instances.append(ImbalancedProblem(spread=1.0).from_generator(51000))  # Running for longer than the simulation time, so we do not run out of tasks
print(Reporter.aggregate(Simulator.replicate(problem_instances, GreedyPlanner(), Reporter(10000), 50000)))
print(Reporter.aggregate(Simulator.replicate(problem_instances, HeuristicPlanner(), Reporter(10000), 50000)))
print(Reporter.aggregate(Simulator.replicate(problem_instances, PredictivePlanner(ImbalancedPredicter), Reporter(10000), 50000)))
print(Reporter.aggregate(Simulator.replicate(problem_instances, PredictivePlanner(PerfectPredicter), Reporter(10000), 50000)))

# Comparing two spreads of the imbalanced problem in a box plot
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
results = dict()
for spread in np.arange(0.5, 1.01, 0.05):
    problem_instances = []
    for i in range(5):
        problem_instances.append(ImbalancedProblem(spread=spread).from_generator(51000))  # Running for longer than the simulation time, so we do not run out of tasks
    result = Reporter.aggregate(Simulator.replicate(problem_instances, HeuristicPlanner(), Reporter(10000), 50000))
    results[spread] = result['task wait time']
line_with_ci(results)

# Printing a trace for the sequential problem
problem_instance = SequentialProblem().from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
reporter = Reporter(warmup=0, reporters=[EventLogReporterElement("temp/my_log.csv", TimeUnit.MINUTES)])
simulator = Simulator(problem_instance, reporter, GreedyPlanner())
simulator.simulate(1000)

# TODO: documentation
# TODO: design an experiment to draw conclusions
# TODO: prepare for cluster computer and run many experiments
# TODO: also test the experiment for a real dataset
