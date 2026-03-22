import numpy as np
import mean_squared_error as mse
def train_linear_regression(X, y, lr=0.01, epochs=1000):
    """
    X: matrix of features (list of lists), one row per match
    y: actual values (list), one per match
    lr: learning rate (step size)
    epochs: how many times to loop through the update

    Returns: learned weights
    """
    # Step 1: Add a bias column to X
    #   (A column of all 1s prepended to X. This gives the model
    #    a "base prediction" even when all features are zero.
    #    Think of it as the y-intercept in y = mx + b)
    
    # Step 2: Initialize weights randomly (small random numbers)
    
    # Step 3: Loop for 'epochs' iterations:
    #   a. Compute predictions (matrix-vector multiply)
    #   b. Compute errors (predictions - actual)
    #   c. Compute gradient: (2/n) * X_transposed · errors
    #   d. Update weights: weights = weights - lr * gradient
    #   e. Every 100 epochs, print the MSE so you can watch it decrease
    
    # Step 4: Return the learned weights

    #Step 1
    #Create new matrix by adding ones to each of the value in the list
    X_b = np.column_stack([np.ones(len(X)), X])
    print(f"X: {X_b}")

    #Step 2
    weights = np.ones(len(X_b[1])) * 0.2
    print(f"Initial weights: {weights}")

    #Step 3
    X_b_transposed = np.transpose(X_b)
    epoch = 1
    while epoch <= epochs:
        predictions = np.dot(X_b, weights) # matrix_vector_multiply
        error = [x-y for x,y in zip(predictions, y)]
        scaling_factor = 2/(len(X_b))
        gradient = scaling_factor * np.dot(X_b_transposed,error)
        weights = weights - (lr * gradient)
        if epoch%10 == 0:
            print(f"Epoch {epoch}")
            print(mse.mse(predictions, y))
        epoch += 1
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

final_weights = train_linear_regression(X, y)

final_weights = [x.item() for x in final_weights]
print(final_weights)


"""
Run it and tell me three things:

This is the answer I got for the test values you had for 1000 epochs - [-0.29467793210090043, 0.789013634494123, 0.10444532033534781, -0.9537335727262956]
1. Does MSE decrease over epochs? (It should — if not, your learning rate might be too high) - yes, it does.
2. What weights did it learn? Do they make intuitive sense? - [-0.29467793210090043, 0.789013634494123, 0.10444532033534781, -0.9537335727262956]. It does make intuitive sense, it gives importance to the home goals, it away goals are not given as much importance, least importance is given to their difference.
3. What happens if you change the learning rate to 1.0? What about 0.0001? - When lr is 1, it goes into a scala power overflow and the results into inf error and when it is 0.001 , the error rate a still a bit higher than we had before when it was 0.01

lr = 1.0 → overflow. The steps were so big that the weights shot past the valley, overshot even harder on the next step, and spiralled into infinity. This is called divergence. In the hill analogy, you took such a giant step downhill that you launched yourself over the valley and up the other mountain, then kept bouncing harder each time.
lr = 0.0001 → still high MSE. The model was learning, just extremely slowly. It needed way more than 1000 epochs to get to the bottom. Each step was so tiny it barely moved. Not broken — just slow.
lr = 0.01 → just right. Big enough to make progress, small enough to not explode. Finding this sweet spot is one of the most practical skills in ML. There's no formula — you experiment.
"""