from problems import ImbalancedProblem
from simulator import Simulator
from planners import GreedyPlanner, HeuristicPlanner, PredictivePlanner
from predicters import Predicter, ImbalancedPredicter, PerfectPredicter

problem_instances = []
for i in range(100):
    problem_instance = ImbalancedProblem.from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
    problem_instances.append(problem_instance)
print(Simulator.replicate(problem_instances, GreedyPlanner(), 10000, 50000))
print(Simulator.replicate(problem_instances, HeuristicPlanner(), 10000, 50000))
print(Simulator.replicate(problem_instances, PredictivePlanner(ImbalancedPredicter), 10000, 50000))
print(Simulator.replicate(problem_instances, PredictivePlanner(PerfectPredicter), 10000, 50000))
