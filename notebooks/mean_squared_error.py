from functools import reduce
"""
 Mean Squared Error.
    1. Subtract each prediction from actual (errors)
    2. Square each error
    3. Take the mean
"""
def mse(predictions, actual):
    error = [x-y for x,y in zip(predictions, actual)]
    squared = [x**2 for x in error]
    total = reduce(lambda a,b: a+b, squared)
    mean = total/len(squared)
    return(mean)

predictions = [1.34, -0.05, 3.23, 0.80, 2.10]
actual = [1, 0, 3, 1, 2]
#print(mse(predictions, actual))  # should be approximately 0.0442

"""
Models care about big errors dispropotianately, if there is a huge error of say 5 that happens sometimes but 
if there is an error of 0.5 that happens often, still the one that happens rarely is worst than the later

When there is an error found using mse, it does not tell the model to recheck the weights in any way, it tells 
the model which direction to adjust the weights in and by how much. If loss goes down after the adjustment, it 
went the right way and if goes up, it went the wrong way. This process of adjustment is done using 
GRADIENT DESCENT.

"""