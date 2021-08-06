# # Simulating several planners and predicters for the imbalanced problem
# from problems import ImbalancedProblem
# from simulator import Simulator
# from planners import GreedyPlanner, HeuristicPlanner, PredictivePlanner
# from predicters import ImbalancedPredicter, PerfectPredicter
#
# problem_instances = []
# for i in range(100):
#     problem_instance = ImbalancedProblem.from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
#     problem_instances.append(problem_instance)
# print(Simulator.replicate(problem_instances, GreedyPlanner(), 10000, 50000))
# print(Simulator.replicate(problem_instances, HeuristicPlanner(), 10000, 50000))
# print(Simulator.replicate(problem_instances, PredictivePlanner(ImbalancedPredicter), 10000, 50000))
# print(Simulator.replicate(problem_instances, PredictivePlanner(PerfectPredicter), 10000, 50000))

# # Printing a trace for the sequential problem
# from problems import SequentialProblem
# from simulator import Simulator, Reporter
# from planners import GreedyPlanner
#
# problem_instance = SequentialProblem.from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
# reporter = Reporter(verbose=True)
# simulator = Simulator(problem_instance, reporter, GreedyPlanner())
# simulator.simulate(100)

# Simulating the greedy planner for the sequential problem
from problems import SequentialProblem
from simulator import Simulator
from planners import GreedyPlanner, HeuristicPlanner, PredictivePlanner
from predicters import ImbalancedPredicter, PerfectPredicter

problem_instances = []
for i in range(100):
    problem_instance = SequentialProblem.from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
    problem_instances.append(problem_instance)
print(Simulator.replicate(problem_instances, GreedyPlanner(), 10000, 50000))
print(Simulator.replicate(problem_instances, HeuristicPlanner(), 10000, 50000))
print(Simulator.replicate(problem_instances, PredictivePlanner(ImbalancedPredicter), 10000, 50000))
print(Simulator.replicate(problem_instances, PredictivePlanner(PerfectPredicter), 10000, 50000))

# TODO: make a more general reporter by passing it functions that are handlers for particular reported events; also functions to summarize in the end
# TODO: one of the reporters should be for process mining
# TODO: make a general graph drawing module for reporting particular reported variables [as functions of other variables] [with boxplot/confidence over the replications]
# TODO: design an experiment to draw conclusions
# TODO: prepare for cluster computer and run many experiments
# TODO: also test the experiment for a real dataset
