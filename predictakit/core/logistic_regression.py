import numpy as np

class LogisticRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y, lr=0.01, epochs=1000):
        """
        Train logistic regression with gradient descent.
        
        Almost identical to linear regression except:
        1. predictions go through sigmoid
        2. gradient is (1/n) * X^T · (probabilities - y)
        """
        # Add bias column
        # Initialize weights
        # Loop:
        #   z = X · weights
        #   probabilities = sigmoid(z)
        #   gradient = (1/n) * X^T · (probabilities - y)
        #   weights = weights - lr * gradient
        #   Every 100 epochs: print cross-entropy loss
        X_b = np.column_stack([np.ones(len(X)), X]) # Add bias column
        print(f"X_b {X_b}")
        weights = np.ones(len(X_b[1])) * 0.2 # Add default weights
        print(f"Initial weights {weights}")
        epoch = 1
        while epoch <= epochs:
            z = np.dot(X_b, weights)
            probabilities = self.sigmoid(z)
            gradient = self.gradient(X_b, probabilities, y)
            weights = weights - lr * gradient
            epoch += 1
            self.weights = weights
            if epoch%10 == 0:
                loss = self.binary_cross_entropy(probabilities, y)
                print(f"Epoch: {epoch} Loss: {loss}")

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def gradient(self, X, probabilities, y):
        return (1/len(X) * (np.dot(np.transpose(X), (probabilities - y))))
    
    def binary_cross_entropy(self, predictions, actual):
        total_loss = 0
        for x in range(0,len(predictions)):
            y = actual[x]
            p = p = np.clip(predictions[x], 1e-15, 1 - 1e-15) # This avoids the negative infinity value you get at log(0) and 
            total_loss += -1 * ((y * np.log(p)) + ((1 - y) * np.log(1-p)))
        return total_loss/len(predictions)

    def predict_proba(self, X):
        """Return probabilities."""
        X_b = np.column_stack([np.ones(len(X)), X])
        z = np.dot(X_b, self.weights)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Return 0 or 1 based on threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)