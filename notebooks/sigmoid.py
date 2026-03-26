import numpy as np

def sigmoid(z):
    """
    Takes any number (or array of numbers).
    Returns a value between 0 and 1.

    sigmoid - σ(z) = 1 / (1 + e^(-z))
    """
    # Your implementation
    sigmoid = 1 / (1 + np.e**(-z))
    return sigmoid

test_values = [-5.0, -2.0, 0.0, 2.0, 5.0]
for z in test_values:
    print(f"z = {z:5.1f} → σ(z) = {sigmoid(z):.4f}")