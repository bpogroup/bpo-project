# # Simulating several planners and predicters for the imbalanced problem
# from problems import ImbalancedProblem
# from simulator import Simulator, Reporter
# from planners import GreedyPlanner, HeuristicPlanner, PredictivePlanner
# from predicters import ImbalancedPredicter, PerfectPredicter
#
# problem_instances = []
# for i in range(100):
#     problem_instance = ImbalancedProblem.from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
#     problem_instances.append(problem_instance)
# print(Simulator.replicate(problem_instances, GreedyPlanner(), Reporter(10000), 50000))
# print(Simulator.replicate(problem_instances, HeuristicPlanner(), Reporter(10000), 50000))
# print(Simulator.replicate(problem_instances, PredictivePlanner(ImbalancedPredicter), Reporter(10000), 50000))
# print(Simulator.replicate(problem_instances, PredictivePlanner(PerfectPredicter), Reporter(10000), 50000))

# # Printing a trace for the sequential problem
# from problems import SequentialProblem
# from simulator import Simulator, Reporter, EventLogReporterElement, TimeUnit
# from planners import GreedyPlanner
#
# problem_instance = SequentialProblem.from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
# reporter = Reporter(warmup=0, reporters=[EventLogReporterElement("temp/my_log.csv", TimeUnit.MINUTES)])
# simulator = Simulator(problem_instance, reporter, GreedyPlanner())
# simulator.simulate(1000)

# # Simulating the greedy planner for the sequential problem
# from problems import SequentialProblem
# from simulator import Simulator, Reporter
# from planners import GreedyPlanner, HeuristicPlanner, PredictivePlanner
# from predicters import ImbalancedPredicter, PerfectPredicter
#
# problem_instances = []
# for i in range(100):
#     problem_instance = SequentialProblem.from_generator(51000)  # Running for longer than the simulation time, so we do not run out of tasks
#     problem_instances.append(problem_instance)
# print(Simulator.replicate(problem_instances, GreedyPlanner(), Reporter(10000), 50000))
# print(Simulator.replicate(problem_instances, HeuristicPlanner(), Reporter(10000), 50000))
# print(Simulator.replicate(problem_instances, PredictivePlanner(ImbalancedPredicter), Reporter(10000), 50000))
# print(Simulator.replicate(problem_instances, PredictivePlanner(PerfectPredicter), Reporter(10000), 50000))

# TODO: make a general graph drawing module for reporting particular reported variables [as functions of other variables] [with boxplot/confidence over the replications]
# TODO: documentation
# TODO: design an experiment to draw conclusions
# TODO: prepare for cluster computer and run many experiments
# TODO: also test the experiment for a real dataset
