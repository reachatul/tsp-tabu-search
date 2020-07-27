#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import os
import numpy as np
# import multiprocessing
# os.getcwd()




Point = namedtuple("Point", ['x', 'y'])


file_location = 'TSP/data//tsp_51_1'
with open(file_location, 'r') as input_data_file:
    input_data = input_data_file.read()



def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)





def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # z = np.array([complex(point.x, point.y) for point in points])
    # m, n = np.meshgrid(z, z, sparse=True)
    # distance = abs(m-n)

    def create_initial_feasible_solution(nodeCount, method):
        solution = [0]
        nodes = list(range(1, nodeCount))
        if method == "nearest_neighbourhood":
            while nodes:
                nearest_index = np.argmin([length(points[solution[-1]], points[k]) for k in nodes])
                # print(nodes)
                # print([length(points[0], points[k]) for k in nodes])
                solution.append(nodes[nearest_index])
                del nodes[nearest_index]
            return solution


    sol = create_initial_feasible_solution(nodeCount, 'nearest_neighbourhood')

    import random

    def move(a, b, sol):
        sol[b], sol[a] = sol[a], sol[b]
        return sol




    def difference_swap(solution, a, b):
        # convert solution into a hamiltonian cycle
        sol = solution + [0]
        if b - a == 1:
            return ((length(points[sol[a-1]], points[sol[a]]) +
                    length(points[sol[b]], points[sol[b+1]]) )
            -
            (length(points[sol[a-1]], points[sol[b]]) +
                    length(points[sol[a]], points[sol[b+1]])))

        elif a - b == 1:
            return ((length(points[sol[b-1]], points[sol[b]]) +
                    length(points[sol[a]], points[sol[a+1]]) )
            -
            (length(points[sol[b-1]], points[sol[a]]) +
                    length(points[sol[b]], points[sol[a+1]])))

        else:
            return ((length(points[sol[a-1]], points[sol[a]]) +
            length(points[sol[a]], points[sol[a+1]]) +
                length(points[sol[b-1]], points[sol[b]]) +
                    length(points[sol[b]], points[sol[b+1]])) -
            (length(points[sol[a-1]], points[sol[b]]) +
                length(points[sol[b]], points[sol[a+1]]) +
                    length(points[sol[b-1]], points[sol[a]]) +
                        length(points[sol[a]], points[sol[b+1]])))




    def path_length(sol):
        return sum([length(points[i], points[j]) for i, j in zip(sol, sol[1:] + [0])])




    def tabu_search(iterations):
        tabu_tenure =  np.array([[0 for i in range(nodeCount)] for j in range(nodeCount)])
        sol = create_initial_feasible_solution(nodeCount, 'nearest_neighbourhood')
        print(path_length(sol))
        for iter in range(iterations):
            insert = False
            a, b = sorted(random.sample(range(1, nodeCount), 2))
            swap_difference = difference_swap(sol, a, b)
            insert_difference = difference_insert(sol, a, b)
            if not tabu_tenure[a, b]:
                if swap_difference >= 0 and swap_difference >= insert_difference:
                    print("sdifference is {}".format(swap_difference))
                    # previous_solution = path_length(sol)
                    sol = move_swap(sol, a, b)
                    # print(previous_solution-path_length(sol))
                    tabu_tenure[a, b] = 7
                elif insert_difference > 0:
                    print("idifference is {}".format(insert_difference))
                    # previous_solution = path_length(sol)
                    sol = move_insert(sol, a, b)
                    # print(previous_solution-path_length(sol))
                    tabu_tenure[a, b] = 3
                # elif swap_difference >= -1*(path_length(sol)/nodeCount) and random.random() <= 0.001:
                #     print("bad difference is {}".format(insert_difference))
                #     sol = move_swap(sol, a, b)
                #     tabu_tenure[a, b] += 3.0
                # elif insert_difference >= -1*(path_length(sol)/nodeCount) and random.random() <= 0.001:
                #     print("bad difference is {}".format(insert_difference))
                #     sol = move_insert(sol, a, b)
                #     tabu_tenure[a, b] += 3.0
                else:
                    tabu_tenure[a, b] += 2
                # print(tabu_tenure[a, b])
            else:
                if swap_difference >= 0 and swap_difference >= insert_difference:
                    print("sdifference is {}".format(swap_difference))
                    # previous_solution = path_length(sol)
                    sol = move_swap(sol, a, b)
                    # print(previous_solution-path_length(sol))
                    tabu_tenure[a, b] = 3.0
                elif insert_difference > 0:
                    print("idifference is {}".format(insert_difference))
                    # previous_solution = path_length(sol)
                    sol = move_insert(sol, a, b)
                    # print(previous_solution-path_length(sol))
                    tabu_tenure[a, b] = 2.0
                else:
                    tabu_tenure[a,b] -= 1
        print(path_length(sol))
        print(tabu_tenure)
        return sol, tabu_tenure



    # Swap function
    def move_swap(list, pos1, pos2):
        new_list = list.copy()
        new_list[pos1], new_list[pos2] = new_list[pos2], new_list[pos1]
        return new_list



    def move_insert(list, index1, index2):
        # element from index 2 is put at index 1
        list_new = list.copy()
        list_new.insert(index1, list_new.pop(index2))
        return list_new


    the_list= [0, 1, 2, 3, 4,5, 6, 7 ,8]



    a, b = 5, 1
    move_insert(the_list, a, b)

    # before insert

#     the_list[b-1], the_list[b]
#     the_list[b], the_list[b+1]
# ->  the_list[b-1], the_list[b+1]
#
#     the_list[a-1], the_list[a]
# -> the_list[a-1], the_list[b]
#     the_list[b], the_list[a]

    def difference_insert(sol, a, b):
        sol = sol + [0]
        return ((length(points[sol[b-1]], points[sol[b]]) +
                    length(points[sol[b]], points[sol[b+1]]) +
                    length(points[sol[a-1]], points[sol[a]])) -
                (length(points[sol[b-1]], points[sol[b+1]]) +
                    length(points[sol[a-1]], points[sol[b]]) +
                    length(points[sol[b]], points[sol[a]])))





    solution, tt = tabu_search(1000)

import sys
np.set_printoptions(threshold=sys.maxsize)
tt[0,0] +=1
    path_length(solution)










    def create_data_model():
      """Stores the data for the problem"""
      data = {}
      # Locations in block units
      _locations=[]
      for i in range(1, nodeCount+1):
          line = lines[i]
          parts = line.split()
          _locations.append((float(parts[0]), float(parts[1])))


      # Multiply coordinates in block units by the dimensions of an average city block, 114m x 80m,
      # to get location coordinates.
      data["locations"] = _locations
      data["num_locations"] = nodeCount
      return data


    def manhattan_distance(position_1, position_2):
        """Computes the Manhattan distance between two points"""

        return (
            math.sqrt((position_1[0] - position_2[0])**2 + (position_1[1] - position_2[1])**2))

    def create_distance_callback(data):
        """Creates callback to return distance between points."""


        def distance_callback(from_node, to_node):
          """Returns the manhattan distance between the two nodes"""
          return manhattan_distance(data["locations"][from_node],
                                    data["locations"][to_node])

        return distance_callback

    routing = pywrapcp.RoutingModel(nodeCount, 1, 0)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit_ms = 300000

    data = create_data_model()
    # Create the distance callback.
    dist_callback = create_distance_callback(data)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # calculate the length of the tour
    # obj = length(points[solution[-1]], points[solution[0]])
    # for index in range(0, nodeCount-1):
    #     obj += length(points[solution[index]], points[solution[index+1]])
    if assignment:
      # Solution distance.
      obj = assignment.ObjectiveValue()
      # Display the solution.
      # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
      route_number = 0
      index = routing.Start(route_number) # Index of the variable for the starting node.
      solution = []
      while not routing.IsEnd(index):
        # Convert variable indices to node indices in the displayed route.
        solution.append(routing.IndexToNode(index))
        index = assignment.Value(routing.NextVar(index))
      solution.append(routing.IndexToNode(index))
    else:
      print('No solution found.')

    obj = 0
    i_ = 0
    for i in solution[1:]:
        obj = obj + manhattan_distance(data["locations"][i_],
                           data["locations"][i])
        i_ = i

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution[:-1]))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
