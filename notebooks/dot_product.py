import numpy

"""
Takes two lists of numbers (same length).
Multiplies element by element.
Sums the results.
Returns a single number.

Example:
    match   = [2, 14, 6,  1,   8,   3]
    weights = [0.45, 0.05, 0.15, -0.40, -0.05, -0.12]
"""

def dot_product(match,weights):
    total = 0
    for i in range(0, len(match)):
        total += match[i] * weights[i]
    return total

def dot_product_numpy(match, weights):
    return numpy.dot(match, weights)

match   = [2, 14, 6,  1,   8,   3]
weights = [0.45, 0.05, 0.15, -0.40, -0.05, -0.12]
result = dot_product(match, weights)
print(result)

result_numpy = dot_product_numpy(match, weights)
print(result_numpy)

""" 
The above match describes a football match score line and the weights show the importance the model gives 
to each of this property.
Now if we increase one value, like we change 0.45 to 0.90, this mean the model will find the 1st feature to have
more importance and it will increase the dot product as well. This is called `feature importance`

"""