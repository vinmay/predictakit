import numpy as np
from functools import reduce

class LinearRegression:
    def __init__(self):
        self.weights = None

    def mse(self, predictions, actual):
        error = [x-y for x,y in zip(predictions, actual)]
        squared = [x**2 for x in error]
        total = reduce(lambda a,b: a+b, squared)
        mean = total/len(squared)
        return(mean)

    def fit(self, X, y, method="normal", lr=0.01, epochs=1000):
        """
        Train the model.
        method: normal for normal equation, gradient for gradient descent
        """
        if method == "normal":
            self.weights = self.train_method_normal(X, y)
        else:
            self.weights = self.train_method_gradient(X, y, lr, epochs)
            
    def train_method_normal(self, X, y):
        X_b = np.column_stack([np.ones(len(X)), X])
        X_b_transpose = np.transpose(X_b)
        weights = np.dot(np.linalg.inv(np.dot(X_b_transpose, X_b)) , (np.dot(X_b_transpose, y)))
        return weights
    
    def train_method_gradient(self, X, y, lr, epochs):
        X_b = np.column_stack([np.ones(len(X)), X]) # Add bias column
        weights = np.ones(len(X_b[1])) * 0.2 # Add default weights
        X_b_transposed = np.transpose(X_b) # Transpose X
        epoch = 1
        while epoch <= epochs:
            predictions = np.dot(X_b, weights) # matrix_vector_multiply
            error = [x-y for x,y in zip(predictions, y)] # Calculate error
            scaling_factor = 2/(len(X_b))
            gradient = scaling_factor * np.dot(X_b_transposed,error) # Calculate gradient
            weights = weights - (lr * gradient) # Update weights
            epoch += 1
        return weights
    
    def predict(self, X):
        """
        Make predictions on new data.
        """
        X_b = np.column_stack([np.ones(len(X)), X])
        return (np.dot(X_b, self.weights))