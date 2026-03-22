import dot_product as dp
import numpy

"""
Takes a matrix (list of lists) and a vector (list).
    Returns a list where each element is the dot product
    of one row of the matrix with the vector.

"""

def matrix_vector_multiply(matches, weights):
    result = []
    for i in range(0, len(matches)):
        result.append(dp.dot_product(matches[i], weights))
    return result

def matrix_vector_multiply_numpy(matches, weights):
    return numpy.dot(matches, weights)

matches = [
    [2, 14, 6, 1, 8, 3],
    [1, 9, 4, 1, 11, 5],
    [3, 18, 9, 0, 5, 1],
]
weights = [0.45, 0.05, 0.15, -0.40, -0.05, -0.12]

predictions = matrix_vector_multiply(matches, weights)
print(predictions)
predictions = matrix_vector_multiply_numpy(matches, weights)
print(predictions)

"""
The above implementation is nothing but `forward pass` where we multiple the data with the weights and get 
predictions.

dot_product and matrix_vector_multiply both result is building this forward pass in ML
"""