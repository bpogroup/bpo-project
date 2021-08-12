

def simulation_model_miner(log):
    """
    Mines a simulation model and returns it as a :class:`.Problem` that can be simulated.
    The log from which the model is mined must at least have the columns
    Case ID, Activity, Resource, Start Timestamp, Complete Timestamp,
    which identify the corresponding event log information.

    :param log: a pandas dataframe from which the problem must be mined.
    :return: a :class:`.Problem`.
    """

    # Mine the task types
    # Mine the resources
    # Mine the initial task type distribution
    # Mine the resource pool per task type
    # Mine the processing time distribution per task_type/resource combination
    # Mine the interarrival time
    # Mine the next task type distribution per task type
    # TODO: Data distribution is empty for now, future work
    pass
