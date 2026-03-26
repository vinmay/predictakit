import numpy as np

def binary_cross_entropy(predictions, actual):
    """
    predictions: array of probabilities (between 0 and 1)
    actual: array of 0s and 1s
    
    Returns: average cross-entropy loss (single number)
    
    Hint: np.log() is the natural logarithm.
    Be careful with log(0) — it's negative infinity.
    Add a tiny epsilon (1e-15) to avoid this.
    """
    total_loss = 0
    for x in range(0,len(predictions)):
        y = actual[x]
        p = p = np.clip(predictions[x], 1e-15, 1 - 1e-15) # This avoids the negative infinity value you get at log(0) and 
        total_loss += -1 * ((y * np.log(p)) + ((1 - y) * np.log(1-p)))
    return total_loss/len(predictions)

# Model is very confident and correct
# print(binary_cross_entropy([0.99], [1]))  # should be ~0.01

# # Model is very confident and wrong
# print(binary_cross_entropy([0.01], [1]))  # should be ~4.6

# # Model has no idea
# print(binary_cross_entropy([0.5], [1]))   # should be ~0.69

print(binary_cross_entropy([0.99, 0.01, 0.5], [1, 1, 1])) # should be ~1.77