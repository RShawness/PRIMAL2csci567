from build import cbs_py
import numpy as np
import random
import time 

start_time = time.time()
cols = 10
rows = 60
total_map_size = cols*rows
map = [0 for i in range(total_map_size)]
num_obstacle = 0
while num_obstacle < 50:
    randLoc = int(random.random()*total_map_size)
    if not map[randLoc]:
        map[randLoc] = True
        num_obstacle += 1
agents = 15
startL = []
startD = []
goalL = []
for i in range(agents):
    start_found = False
    goal_found = False
    while not start_found and not goal_found:
        randLoc1 = int(random.random()*total_map_size)
        randLoc2 = int(random.random()*total_map_size)
        if not map[randLoc1] and not map[randLoc2] and randLoc1 not in startL and randLoc2 not in startD:
            startL.append(randLoc1)
            startD.append(int(random.random()*3))
            goalL.append(randLoc2)
            start_found = True
            goal_found = True

# map = [False for i in range(25)]
# map[5] = True
# map[7] = True
# map[9] = True
# map[10] = True
# map[12] = True
# map[14] = True
# map[16] = True
# map[17] = True
# map[19] = True
# agents = 3
# startL = [0,11,15]
# startD = [2,1,3]
# goalL = [24,4,8]
time_limit = 50

print(type(map))
print(type(startL))
print(type(startD))
print(type(goalL))
print(type(cols))
print(type(rows))
print(type(agents))
print(type(time_limit))

result_temp = cbs_py.findPath_new(map, startL, startD, goalL, cols, rows, agents, time_limit)

# if len(result_temp) != 0:
#     result = np.array(result_temp)
#     max_time = np.amax([len(sequence) for sequence in result])
#     transposed_result = [[] for i in range(max_time)]
#     for i in range(max_time):
#         for j in range(len(result)):
#             try: 
#                 transposed_result[i].append(result[j][i])
#             except Exception:
#                 continue 
        
#     print(transposed_result)
#     print(len(transposed_result[0]))
# else:
#     print("no solution found, return None")

print(time.time() - start_time)