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



    def get_neighbourhood(a):
        if random.random() <= 0.70:
            left_, right_ = int(max(1, a-nodeCount/10)), int(min(a+nodeCount/10, nodeCount))
        else:
            left_, right_ = int(max(1, a-nodeCount/5)), int(min(a+nodeCount/5, nodeCount))
        return random.randint(left_, right_)




    def tabu_search(iterations):
        tabu_tenure =  np.array([0 for i in range(nodeCount)])
        sol = create_initial_feasible_solution(nodeCount, 'nearest_neighbourhood')
        print(path_length(sol))
        for iter in range(iterations):
            insert = False
            a, b = random.sample(range(1, nodeCount), 2)
            swap_difference = difference_swap(sol, a, b)
            insert_difference = difference_insert(sol, a, b)

            if iter%500 == 0:
                # diversify
                # for tries in range(int(nodeCount/20)):
                a, b = random.sample(range(1, nodeCount), 2)
                sol = move_swap(sol, a, b)

            if not tabu_tenure[a] or not tabu_tenure[b]:
                if swap_difference >= 0 and swap_difference >= insert_difference:
                    print("sdifference is {}".format(swap_difference))
                    previous_solution = path_length(sol)
                    sol = move_swap(sol, a, b)
                    print(previous_solution-path_length(sol))
                    tabu_tenure[a] = 7
                    tabu_tenure[b] = 7
                elif insert_difference > 0:
                    print(a, b)
                    print("idifference is {}".format(insert_difference))
                    previous_solution = path_length(sol)
                    sol = move_insert(sol, a, b)
                    print(previous_solution-path_length(sol))
                    tabu_tenure[a] = 7
                else:
                    tabu_tenure[a] += 2
                    tabu_tenure[b] += 2
                # print(tabu_tenure[a, b])
            else:
                if swap_difference >= 0 and swap_difference >= insert_difference:
                    # print("sdifference is {}".format(swap_difference))
                    # previous_solution = path_length(sol)
                    sol = move_swap(sol, a, b)
                    # print(previous_solution-path_length(sol))
                    tabu_tenure[a] = 3.0
                    tabu_tenure[b] = 3.0
                elif insert_difference > 0:
                    # print("idifference is {}".format(insert_difference))
                    # previous_solution = path_length(sol)
                    sol = move_insert(sol, a, b)
                    # print(previous_solution-path_length(sol))
                    tabu_tenure[a] = 2.0
                    tabu_tenure[b] = 2.0
                else:
                    tabu_tenure[a] -= 1
                    tabu_tenure[b] -= 1
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

    a, b = 1, 5
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
        if b > a:
            return ((length(points[sol[b-1]], points[sol[b]]) +
                    length(points[sol[b]], points[sol[b+1]]) +
                    length(points[sol[a-1]], points[sol[a]])) -
                (length(points[sol[b-1]], points[sol[b+1]]) +
                    length(points[sol[a-1]], points[sol[b]]) +
                    length(points[sol[b]], points[sol[a]])))
        if a > b:
            return ((length(points[sol[b-1]], points[sol[b]]) +
                    length(points[sol[b]], points[sol[b+1]]) +
                    length(points[sol[a-1]], points[sol[a]])) -
                (length(points[sol[b-1]], points[sol[b+1]]) +
                    length(points[sol[a]], points[sol[b]]) +
                    length(points[sol[b]], points[sol[a+1]])))





    solution, tt = tabu_search(999)

    class TabuSearch(object):
        """docstring for TabuSearch."""

        def __init__(self, nodeCount):

            self.nodeCount = nodeCount


        def create_initial_feasible_solution(self, method):
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

        def move(self, a, b, sol):
            sol[b], sol[a] = sol[a], sol[b]
            return sol


    ts = TabuSearch(51)
    solution = ts.create_initial_feasible_solution("nearest_neighbourhood")



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
