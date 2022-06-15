import numpy as np
from math import *
from fractions import Fraction
import copy

def norm(v):
    answer = np.linalg.norm(v)
    return answer

def linearcombo(u,v,w):
    return 3/2 * u - w + v

def dotproduct(u, v) :
    vector1 = u
    vector2 = v
    return vector1.dot(vector2)

v = np.array([0, 5])
u = np.array([2, -1])
t = np.array([-3, 4])
### finding t as combination of u and v
def basis_vector_thingy(t, u, v):
    a = Fraction(dotproduct(u, t)/dotproduct(u, u)).limit_denominator(100)
    b = Fraction(dotproduct(v, t)/dotproduct(v, v)).limit_denominator(100)
    print(f't = {a} * u + {b} * v')

def mag_and_dir_to_components(mag, angle):
   print(f'{mag * cos(radians(angle))/sqrt(2)}i + {mag * sin(radians(angle))/sqrt(2)}j')

def find_parametric(P, v):
    print(f'x = {v[0]}t + {P[0]}')
    print(f'y = {v[1]}t + {P[1]}')

def work(force, angle_dist, distance, angle_force):
    print(force * distance * cos(radians(angle_dist-angle_force)))

A = [[-7, 3],
     [5, 2]]

B = [[6, -1],
     [9, -8]]

C = [[8, 6, -1],
     [0, -2, 5],
     [4, 3, 1]]

D = [[14, -3, 2],
     [-4, 0 , 9],
     [-2, 0, 6]]

E = [[5, 8, -3],
     [-6, 0, -4]]

F = [[2, 4, 1],
     [-1, -3, 0]]

G = [[2, -1],
     [3, 5],
     [-1, 6]]
def matrix_add(m1, m2):
    if len(m1) != len(m2) or len(m1[0]) != len(m2[0]):
        raise ValueError("m1 and m2 don't have the same dimesions")
    final_matrix = []
    l = []
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            value = m1[i][j] + m2[i][j]
            l.append(value)
        final_matrix.append(l)
        l = []
    return final_matrix

def mat_mul(scalar=None,m1=None, m2=None):
    if scalar and m1:
        new_row = []
        new_matrix = []
        for row in m1:
            for val in row:
                new_val = val * scalar
                new_row.append(new_val)
            new_matrix.append(new_row)
            new_row = []
        return new_matrix

    else:
        if len(m1[0]) != len(m2):
            raise ValueError("The dimensions don't work for multiplication")
        columns = []
        column = []
        for i in range(len(m2[0])):
            for j in range(len(m2)):
                value = m2[j][i]
                column.append(value)
            columns.append(column)
            column = []

        rows = m1
        ## dotproduct
        final_matrix = []
        final_row = []
        value = 0
        for row in rows:
            for column in columns:
                for i in range(len(row)):
                    value += row[i] * column[i]
                final_row.append(value)
                value = 0
            final_matrix.append(final_row)
            final_row = []

        for row in final_matrix:
            # for i in range(len(row)):
            #     row[i] = round(row[i])
            print(row)

        return final_matrix

               ## Put whatevever expression you need for matrix operations down here.
               ## Matrices can be changed above

A = [[11, 24, 34],
     [23, 13, 94],
     [13, 21, 12]]

B =  [[-0.136, 0.032, 0.136],
      [0.071, -0.023, -0.019],
      [0.023, 0.006, -0.031]]
final_matrix = mat_mul(m1=B,m2=A)

# matrix for test purposes to figure out how to solve it algorithmically, last col is the values.
aug_matrix=[[3, 8, 2, 11],
            [2, 2, 3, 8],
            [-9, 2, 3, -3]]

# m has n rows and n + 1 columns, 2d array, general systems of equations solver (3 var, 4 var, etc.)
def solve_matrix_and_matrix_inverse(m):
    n = len(m)
    for i in range(n):
        row = m[i]
        number = row[i] # number to divide the row by
        for j in range(len(row)):
            value = row[j]
            row[j] = value/number
        copy_ = copy.deepcopy(m)
        copy_.remove(row)

        for row_ in copy_:
            multiplier = row_[i]/row[i]
            for j in range(len(row_)):
                row_[j] -= multiplier * row[j]
        copy_.insert(i, row)
        m = copy_
    for row in m:
        for i in range(len(row)):
            row[i] = round(row[i], 3)
        print(row)

## augment the identity matrix if u want the inverse matrix; augment a whole column if u want to find solutions to system of eqn
aug_matrix=[[10, 24, 34, 1, 0, 0],
            [23, 13, 94, 0, 1, 0],
            [13, 21, 12, 0, 0, 1]]

solve_matrix_and_matrix_inverse(aug_matrix)