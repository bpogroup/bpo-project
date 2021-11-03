import pandas as pd

# We need to clean the dataset. There are several challenges:
# - the tasks use lifecycles: they are started, completed, suspended, resumed we need to aggregate each task start-resume/complete-suspend combination to a single line
# - task complete events are actually single events with a start time and a complete time, but they may be suspended and resumed in between
# - there are tasks complete without explicitly starting: remove cases that have that
# - there are tasks that start twice: remove cases that have that
# After doing all that, we have a dataset that has about 95% of the cases left.
# All tasks now start and complete, so we have their duration.

# %% First we need to expand the complete events

rhandle = open("BPI Challenge 2017.csv", "r")
whandle = open("BPI Challenge 2017 - lifecycle expanded.csv", "w")

whandle.write(rhandle.readline())  # write the header row

for line in rhandle:
    data = line.split(",")
    if data[7] == "complete":  # if this is a complete action, we must split it up
        row1 = data.copy()
        row1[7] = "start"  # the first is a start event 
        row1[4] = row1[3]  # with the completion time set to the start time
        row2 = data.copy()
        row2[3] = row2[4]  # the second is a complete event with the start time set to the completion time
        whandle.write(','.join(row1))
        whandle.write(','.join(row2))
    elif data[7] in ['start', 'suspend', 'resume']:
        whandle.write(','.join(data))
whandle.flush()
whandle.close()
rhandle.close()


# %% Now we sort the file

log = pd.read_csv("BPI Challenge 2017 - lifecycle expanded.csv")
log = log.sort_values(by=['Case ID', 'Start Timestamp'])
log.to_csv("BPI Challenge 2017 - sorted.csv", index=False)


# %% Now we merge (start|resume) / (complete|suspend) combinations

rhandle = open("BPI Challenge 2017 - sorted.csv", "r")
whandle = open("BPI Challenge 2017 - clean.csv", "w")

whandle.write(rhandle.readline())  # write the header row
case = None
started_tasks = dict()  # started tasks are either started or resumed and not yet completed or suspended
nr_cases = 0
problem_case = False
nr_cases_wo_problems = 0
data_to_write = ""
for line in rhandle:
    data = line.split(",")
    if data[0] != case:
        # if this is the next case number   
        nr_cases += 1
        if len(started_tasks) > 0:
            # if there are still started tasks from the previous case: problem
            print("WARNING: case has unfinished tasks", case)
            problem_case = True
        if not problem_case:
            # if the previous case was not problematic: write it to disk
            whandle.write(data_to_write)
            nr_cases_wo_problems += 1
        # start a new case
        problem_case = False 
        started_tasks = dict()
        case = data[0]
        data_to_write = ""
    if data[7] in ['start', 'resume']:                
        # if the line starts a task
        if data[1] in started_tasks.keys():
            # if the task is already started: problem
            # print("WARNING: case has task that starts twice", case)
            problem_case = True    
        # start the task
        started_tasks[data[1]] = (data[3], data[2]) # store start time, resource
    if data[7] in ['suspend', 'complete']:        
        # if the line completes a task:
        if not data[1] in started_tasks.keys():
            # if the task is not started: problem
            # print("WARNING: case has task that completes without starting", case)
            problem_case = True        
        else:    
            # write the task with the corresponding start and completion time
            data[3] = started_tasks[data[1]][0]
            # it happens that tasks are suspended by a different user than the user that started the task
            # that is a bit strange, but we will assume that the user that started the task is the one who performs it
            data[2] = started_tasks[data[1]][1]
            data_to_write += ','.join(data)
            # remove the task from the started tasks
            del started_tasks[data[1]]
# still need to process the last case
if len(started_tasks) > 0:
    # if there are still started tasks: problem
    print("WARNING: case has unfinished tasks", case)
    problem_case = True
if not problem_case:
    # if the case was not problematic: write it to disk
    whandle.write(data_to_write)
    nr_cases_wo_problems += 1
whandle.flush()
whandle.close()
rhandle.close()

print(nr_cases, nr_cases_wo_problems, nr_cases_wo_problems/nr_cases)
