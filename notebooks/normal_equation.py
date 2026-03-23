import gradient_descent as gd
import numpy as np
def train_normal_equation(X, y):
    """
    Computes the exact best weights in one step.
    w = (X^T · X)^(-1) · X^T · y
    
    No learning rate. No epochs. No loop.
    """
    # Step 1: Add bias column (same as before)
    # Step 2: Apply the formula using np.dot and np.linalg.inv
    # Step 3: Return weights
     #Step 1
    #Create new matrix by adding ones to each of the value in the list
    X_b = np.column_stack([np.ones(len(X)), X])
    print(f"X: {X_b}")

    X_b_transpose = np.transpose(X_b)

    weights = np.dot(np.linalg.inv(np.dot(X_b_transpose, X_b)) , (np.dot(X_b_transpose, y)))

    return weights

# 5 matches: [home_goals, home_shots_on_target, away_goals]
X = [
    [2, 6, 1],
    [1, 4, 1],
    [3, 9, 0],
    [0, 2, 2],
    [1, 5, 1],
]

# Target: goal difference (home_goals - away_goals)
y = [1, 0, 3, -2, 0]

final_weights = train_normal_equation(X, y)

print(final_weights)