#!/usr/bin/python
import argparse
import glob
from pathlib import Path
from cbs import CBSSolver
from cbs_3d import CBS3d_Solver
from independent import IndependentSolver
from prioritized import PrioritizedPlanningSolver
from visualize import Animation
from single_agent_planner import get_sum_of_cost

SOLVER = "CBS"

def print_mapf_instance(my_map, starts, goals):
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)

def print_locations(my_map, locations):
    # initialize starts_map with all '-1' corresponding to the dimension of the map (or my_map)
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    # for each agent i: write the agent index into (logical) my_map
    for i in range(len(locations)):
        # get the starting (terminating) location index for each agent i in the my_map list
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    # x: row index in (logical) my_map
    for x in range(len(my_map)):
        # y: column index in (logical) my_map
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            # if true in (logical) my_map print the obstacle mark '@ '
            elif my_map[x][y]:
                to_print += '@ '
            # if false in (logical) my_map print the obstacle mark '. '
            else:
                to_print += '. '
        # add '\n' after processing each line
        to_print += '\n'
    print(to_print)

def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    # get map information
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    # #agents
    line = f.readline()
    num_agents = int(line)
    # #agents lines with the start/goal positions
    # get each agent's start and goal location
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    # my_map: a list of (true and false corresponding to the discrete obstacles and ways in each line of the map)
    # starts(goals): a list of each agent's starting(terminating) cell location in discrete coordination
    return my_map, starts, goals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--instance', type=str, default=None,
                        help='The name of the instance file(s)')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch output instead of animation')
    parser.add_argument('--disjoint', action='store_true', default=False,
                        help='Use the disjoint splitting')
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS,Independent,Prioritized}), defaults to ' + str(SOLVER))

    # args = parser.parse_args()
    # manually set input test instances arguments for debugging the code
    # args = parser.parse_args(["--instance", "custominstances/empty-8-8.map", "--solver", "CBS"])
    # args = parser.parse_args(["--instance", "instances/test_0.txt", "--solver", "CBS"])
    # args = parser.parse_args(["--instance", "instances/exp2_3.txt", "--solver", "CBS"])
    # args = parser.parse_args(["--instance", "instances/test_10.txt", "--solver", "Prioritized"])
    # args = parser.parse_args(["--instance", "instances_3d/test_10.mat", "--solver", "CBS_3d"])
    args = parser.parse_args(["--instance", "instances/test_8.txt", "--solver", "CBS"])

    result_file = open("results.csv", "w", buffering=1)

    for file in sorted(glob.glob(args.instance)):

        print("***Import an instance***")
        my_map, starts, goals = import_mapf_instance(file)
        print_mapf_instance(my_map, starts, goals)

        if args.solver == "CBS":
            print("***Run CBS***")
            solver = CBSSolver(my_map, starts, goals)
            paths = solver.find_solution(args.disjoint)
            # paths = solver.find_solution()
        elif args.solver == "CBS_3d":
            print("***Run CBS_3d***")
            cbs_3d = CBS3d_Solver(my_map, starts, goals)
            paths = cbs_3d.find_solution()
        elif args.solver == "Independent":
            print("***Run Independent***")
            solver = IndependentSolver(my_map, starts, goals)
            paths = solver.find_solution()
        elif args.solver == "Prioritized":
            print("***Run Prioritized***")
            solver = PrioritizedPlanningSolver(my_map, starts, goals)
            paths = solver.find_solution()
        else:
            raise RuntimeError("Unknown solver!")

        cost = get_sum_of_cost(paths)
        result_file.write("{},{}\n".format(file, cost))


        if not args.batch:
            print("***Test paths on a simulation***")
            animation = Animation(my_map, starts, goals, paths)
            # animation.save("output.mp4", 1.0)
            animation.show()
    result_file.close()
